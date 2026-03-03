//! Minimal HTTP and WebSocket server on plain threads: serves the dashboard and
//! streams hub packets to each connected viewer.

use super::frame::{self, Buf};
use super::hub::Hub;
use snn_core::Control;
use std::io::{self, ErrorKind, Read, Write};
use std::net::{IpAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Duration;
use tungstenite::{Error as WsError, Message};

/// Viewers streamed to at once; each one holds a thread.
const MAX_VIEWERS: usize = 16;

const INDEX: &str = include_str!("../../../web/index.html");

macro_rules! font {
    ($name:literal) => {
        (
            concat!("/fonts/", $name, ".woff2"),
            include_bytes!(concat!("../../../web/fonts/", $name, ".woff2")).as_slice(),
        )
    };
}

const FONTS: [(&str, &[u8]); 4] = [
    font!("barlow-400"),
    font!("barlow-500"),
    font!("barlow-condensed-400"),
    font!("barlow-condensed-600"),
];

macro_rules! view {
    ($name:literal) => {
        (
            concat!("/views/", $name, ".js"),
            include_str!(concat!("../../../web/views/", $name, ".js")),
        )
    };
}

/// One script per model, picked by the dashboard from META.
const VIEWS: [(&str, &str); 5] = [
    view!("pop"),
    view!("wta"),
    view!("spncn"),
    view!("pcsnn"),
    view!("pcl"),
];

/// Files compiled into the binary, by request path.
fn asset(path: &str) -> Option<(&'static str, &'static [u8])> {
    if path == "/" || path == "/index.html" {
        return Some(("text/html; charset=utf-8", INDEX.as_bytes()));
    }
    if let Some((_, js)) = VIEWS.iter().find(|(p, _)| *p == path) {
        return Some(("text/javascript; charset=utf-8", js.as_bytes()));
    }
    FONTS
        .iter()
        .find(|(p, _)| *p == path)
        .map(|(_, body)| ("font/woff2", *body))
}

/// Accepts connections forever, one thread each.
pub fn serve(listener: TcpListener, hub: Arc<Hub>, ctl: Arc<Control>) {
    publish_state(&hub, &ctl);
    for stream in listener.incoming().flatten() {
        let (hub, ctl) = (hub.clone(), ctl.clone());
        std::thread::spawn(move || {
            let _ = handle(stream, &hub, &ctl);
        });
    }
}

/// Broadcasts pause/speed/done to every viewer.
pub fn publish_state(hub: &Hub, ctl: &Control) {
    let mut b = Buf::new();
    frame::state(&mut b, ctl.is_paused(), ctl.speed(), ctl.is_done());
    hub.set_state(b.take());
}

fn handle(mut stream: TcpStream, hub: &Hub, ctl: &Control) -> io::Result<()> {
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    let head = peek_head(&stream)?;
    let text = String::from_utf8_lossy(&head);
    let upgrade = text.lines().any(|l| {
        let l = l.to_ascii_lowercase();
        l.starts_with("upgrade:") && l.contains("websocket")
    });
    if upgrade {
        if !upgrade_allowed(&text) {
            return reject(stream, "403 Forbidden");
        }
        if hub.viewers() >= MAX_VIEWERS {
            return reject(stream, "503 Service Unavailable");
        }
        return session(stream, hub, ctl);
    }

    let path = text.split_whitespace().nth(1).unwrap_or("/");
    let path = path.split('?').next().unwrap_or("/");
    let mut consumed = vec![0; head.len()];
    stream.read_exact(&mut consumed)?;
    match asset(path) {
        Some((kind, body)) => {
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: {kind}\r\nContent-Length: {}\r\n\
                 Cache-Control: no-cache\r\nConnection: close\r\n\r\n",
                body.len()
            )?;
            stream.write_all(body)?;
        }
        None => stream.write_all(
            b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        )?,
    }
    stream.flush()
}

fn reject(mut stream: TcpStream, status: &str) -> io::Result<()> {
    let head = format!("HTTP/1.1 {status}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
    stream.write_all(head.as_bytes())?;
    stream.flush()
}

fn header<'a>(head: &'a str, name: &str) -> Option<&'a str> {
    head.lines().skip(1).find_map(|l| {
        let (k, v) = l.split_once(':')?;
        k.trim().eq_ignore_ascii_case(name).then(|| v.trim())
    })
}

/// Whether a `Host` value names this machine by address or as localhost. A
/// page reaching the server through a rebound DNS name fails this.
fn is_local_host(host: &str) -> bool {
    let name = match host.strip_prefix('[') {
        Some(rest) => rest.split(']').next().unwrap_or(""),
        None => host.rsplit_once(':').map_or(host, |(name, _)| name),
    };
    name.eq_ignore_ascii_case("localhost") || name.parse::<IpAddr>().is_ok()
}

/// Browsers always send `Origin` on a WebSocket handshake, so another site's
/// page can't drive training; clients that send none are not browsers.
fn upgrade_allowed(head: &str) -> bool {
    let Some(host) = header(head, "host").filter(|h| is_local_host(h)) else {
        return false;
    };
    header(head, "origin").is_none_or(|origin| {
        origin
            .split_once("://")
            .is_some_and(|(_, authority)| authority.eq_ignore_ascii_case(host))
    })
}

/// The request head, left unread in the socket.
fn peek_head(stream: &TcpStream) -> io::Result<Vec<u8>> {
    let mut buf = [0u8; 8192];
    loop {
        let n = stream.peek(&mut buf)?;
        if n == 0 {
            return Err(ErrorKind::UnexpectedEof.into());
        }
        if let Some(end) = buf[..n].windows(4).position(|w| w == b"\r\n\r\n") {
            return Ok(buf[..end + 4].to_vec());
        }
        if n == buf.len() {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "request head too large",
            ));
        }
        // only part of the head has arrived
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn session(stream: TcpStream, hub: &Hub, ctl: &Control) -> io::Result<()> {
    let mut ws = tungstenite::accept(stream).map_err(|e| io::Error::other(e.to_string()))?;
    // short reads so the loop keeps forwarding packets
    ws.get_mut()
        .set_read_timeout(Some(Duration::from_millis(2)))?;
    let mut viewer = hub.join();
    let ws_err = |e: WsError| io::Error::other(e.to_string());
    for p in viewer.intro() {
        ws.write(Message::Binary(p)).map_err(ws_err)?;
    }
    loop {
        match ws.read() {
            Ok(Message::Text(cmd)) => command(cmd.as_str(), hub, ctl),
            Ok(Message::Close(_)) => return Ok(()),
            Ok(_) => {}
            Err(WsError::Io(e))
                if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
            Err(WsError::ConnectionClosed | WsError::AlreadyClosed) => return Ok(()),
            Err(e) => return Err(ws_err(e)),
        }
        for p in viewer.poll(Duration::from_millis(15)) {
            ws.write(Message::Binary(p)).map_err(ws_err)?;
        }
        ws.flush().map_err(ws_err)?;
    }
}

/// Applies `pause`, `run`, `speed <steps per second>` or `probe <pop> <unit>`
/// (`probe auto` to clear) from a viewer.
fn command(cmd: &str, hub: &Hub, ctl: &Control) {
    let mut it = cmd.split_whitespace();
    match (it.next(), it.next()) {
        (Some("pause"), _) => ctl.set_paused(true),
        (Some("run"), _) => ctl.set_paused(false),
        (Some("speed"), Some(v)) => match v.parse() {
            Ok(v) => ctl.set_speed(v),
            Err(_) => return,
        },
        (Some("probe"), Some("auto")) => return ctl.set_probe(None),
        (Some("probe"), Some(pop)) => {
            if let (Ok(pop), Some(Ok(unit))) = (pop.parse(), it.next().map(str::parse)) {
                ctl.set_probe(Some((pop, unit)));
            }
            return;
        }
        _ => return,
    }
    publish_state(hub, ctl);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::SocketAddr;

    fn start() -> (SocketAddr, Arc<Hub>, Arc<Control>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let hub = Arc::new(Hub::new());
        let ctl = Arc::new(Control::new(1000));
        hub.set_meta(vec![frame::META, 0, 0, 0, 0]);
        let (h, c) = (hub.clone(), ctl.clone());
        std::thread::spawn(move || serve(listener, h, c));
        (addr, hub, ctl)
    }

    fn get(addr: SocketAddr, path: &str) -> String {
        let mut s = TcpStream::connect(addr).unwrap();
        write!(s, "GET {path} HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
        let mut out = Vec::new();
        s.read_to_end(&mut out).unwrap();
        String::from_utf8_lossy(&out).into_owned()
    }

    #[test]
    fn serves_the_page_and_404s() {
        let (addr, _, _) = start();
        let page = get(addr, "/?v=1");
        assert!(page.starts_with("HTTP/1.1 200 OK"));
        assert!(page.contains("text/html"));
        assert!(page.contains("<html"));
        assert!(get(addr, "/missing").starts_with("HTTP/1.1 404"));
        let view = get(addr, "/views/wta.js");
        assert!(view.contains("text/javascript") && view.contains("registerView"));
        let font = get(addr, "/fonts/barlow-400.woff2");
        assert!(font.contains("font/woff2") && font.contains("wOF2"));
    }

    /// Status line (`HTTP/1.1 nnn`) the server answers a handshake from a host and origin with.
    fn upgrade(addr: SocketAddr, host: &str, origin: Option<&str>) -> String {
        let mut s = TcpStream::connect(addr).unwrap();
        let origin = origin.map_or(String::new(), |o| format!("Origin: {o}\r\n"));
        write!(
            s,
            "GET /ws HTTP/1.1\r\nHost: {host}\r\n{origin}Upgrade: websocket\r\n\
             Connection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\
             Sec-WebSocket-Version: 13\r\n\r\n"
        )
        .unwrap();
        let mut status = [0u8; 12];
        s.read_exact(&mut status).unwrap();
        String::from_utf8_lossy(&status).into_owned()
    }

    #[test]
    fn only_addresses_and_localhost_are_local_hosts() {
        for h in [
            "127.0.0.1:7878",
            "localhost:80",
            "LOCALHOST",
            "[::1]:7878",
            "10.0.0.5:9",
        ] {
            assert!(is_local_host(h), "{h}");
        }
        for h in [
            "evil.example:7878",
            "127.0.0.1.evil.com",
            "localhost.evil.com:80",
            "",
        ] {
            assert!(!is_local_host(h), "{h}");
        }
    }

    #[test]
    fn other_sites_cannot_open_a_viewer() {
        let (addr, _, _) = start();
        let me = addr.to_string();
        let evil = format!("evil.example:{}", addr.port());
        for (host, origin) in [
            (&me, Some("http://evil.example")),
            (&me, Some("null")),
            (&evil, Some(format!("http://{evil}").as_str())),
        ] {
            let answer = upgrade(addr, host, origin);
            assert!(
                answer.starts_with("HTTP/1.1 403"),
                "{host} {origin:?}: {answer}"
            );
        }
        let own = upgrade(addr, &me, Some(&format!("http://{me}")));
        assert!(own.starts_with("HTTP/1.1 101"), "{own}");
        assert!(upgrade(addr, &me, None).starts_with("HTTP/1.1 101"));
    }

    #[test]
    fn viewers_are_capped() {
        let (addr, hub, _) = start();
        let me = addr.to_string();
        let held: Vec<_> = (0..MAX_VIEWERS).map(|_| hub.join()).collect();
        assert!(upgrade(addr, &me, None).starts_with("HTTP/1.1 503"));
        drop(held);
        assert!(upgrade(addr, &me, None).starts_with("HTTP/1.1 101"));
    }

    #[test]
    fn websocket_gets_intro_and_controls_training() {
        let (addr, hub, ctl) = start();
        let stream = TcpStream::connect(addr).unwrap();
        let (mut ws, _) = tungstenite::client(format!("ws://{addr}/ws"), stream).unwrap();
        let first = ws.read().unwrap().into_data();
        assert_eq!(first[0], frame::META);
        assert_eq!(ws.read().unwrap().into_data()[0], frame::STATE);

        ws.send(Message::text("pause")).unwrap();
        let state = ws.read().unwrap().into_data();
        assert_eq!(state[0], frame::STATE);
        assert_eq!(state[5], 1, "paused flag");
        assert!(ctl.is_paused());

        ws.send(Message::text("speed 5000")).unwrap();
        ws.read().unwrap();
        assert_eq!(ctl.speed(), 5000);

        ws.send(Message::text("probe 2 17")).unwrap();
        ws.send(Message::text("probe 1")).unwrap();
        ws.send(Message::text("speed 6000")).unwrap();
        ws.read().unwrap();
        assert_eq!(ctl.probe(), Some((2, 17)));

        hub.publish(vec![frame::EPOCH, 0, 0, 0, 0]);
        assert_eq!(ws.read().unwrap().into_data()[0], frame::EPOCH);
        assert_eq!(hub.viewers(), 1);
    }
}
