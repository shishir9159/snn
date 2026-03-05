//! What the network can be shown: still images that saccades turn into events,
//! or recordings an event camera made of the same digits.

use snn_core::events::{Events, Saccades};
use snn_core::nmnist::Recordings;
use snn_core::Dataset;
use snn_live::frame::Param;

pub trait Input {
    fn name(&self) -> &str;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn classes(&self) -> usize;
    fn label(&self, i: usize) -> u8;
    /// Width and height of the picture the dashboard shows, and the retina side
    /// the network needs.
    fn picture(&self) -> (usize, usize);
    fn retina(&self, saccades: &Saccades) -> usize;
    /// Steps a presentation lasts.
    fn steps(&self, saccades: &Saccades) -> usize;
    /// Puts sample `i` into `events` and its picture into `picture`.
    fn load(&self, i: usize, saccades: &Saccades, events: &mut Events, picture: &mut Vec<u8>);
    /// Rows for the dashboard's model panel about where the events come from.
    fn source(&self, saccades: &Saccades) -> Vec<Param<'static>>;
}

impl Input for Dataset {
    fn name(&self) -> &str {
        &self.name
    }

    fn len(&self) -> usize {
        Dataset::len(self)
    }

    fn classes(&self) -> usize {
        self.classes
    }

    fn label(&self, i: usize) -> u8 {
        Dataset::label(self, i)
    }

    fn picture(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    fn retina(&self, s: &Saccades) -> usize {
        self.width.max(self.height) + 2 * s.pad
    }

    fn steps(&self, s: &Saccades) -> usize {
        s.steps()
    }

    fn load(&self, i: usize, s: &Saccades, events: &mut Events, picture: &mut Vec<u8>) {
        let img = self.image(i);
        events.generate(img, self.width, self.height, s);
        picture.clear();
        picture.extend_from_slice(img);
    }

    fn source(&self, s: &Saccades) -> Vec<Param<'static>> {
        vec![
            Param::num("Saccade px", s.amplitude, ""),
            Param::num("Saccade leg", s.leg as f32, "ms"),
        ]
    }
}

impl Input for Recordings {
    fn name(&self) -> &str {
        &self.name
    }

    fn len(&self) -> usize {
        Recordings::len(self)
    }

    fn classes(&self) -> usize {
        self.classes
    }

    fn label(&self, i: usize) -> u8 {
        Recordings::label(self, i)
    }

    fn picture(&self) -> (usize, usize) {
        (snn_core::nmnist::SIDE, snn_core::nmnist::SIDE)
    }

    fn retina(&self, _: &Saccades) -> usize {
        snn_core::nmnist::SIDE
    }

    fn steps(&self, _: &Saccades) -> usize {
        Recordings::steps(self)
    }

    fn load(&self, i: usize, _: &Saccades, events: &mut Events, picture: &mut Vec<u8>) {
        self.fill(i, events);
        self.preview(i, picture);
    }

    fn source(&self, _: &Saccades) -> Vec<Param<'static>> {
        vec![Param::text("Recorded", "event camera".into())]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{train::Trainer, Params};
    use snn_core::dataset::synthetic;
    use snn_core::{nmnist, Control};

    #[test]
    fn recordings_train_like_the_images_they_came_from() {
        let root = std::env::temp_dir().join(format!("snn-pcl-input-{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        let s = Saccades::default();
        nmnist::export(&synthetic(600, 1), &s, &root, "Train").unwrap();
        nmnist::export(&synthetic(200, 2), &s, &root, "Test").unwrap();
        let train = Recordings::load(&root, true, None).unwrap();
        let test = Recordings::load(&root, false, None).unwrap();
        std::fs::remove_dir_all(&root).ok();

        assert_eq!((train.picture(), train.retina(&s)), ((34, 34), 34));
        assert_eq!(
            (
                train.len(),
                test.len(),
                Input::steps(&train, &s) <= s.steps()
            ),
            (600, 200, true)
        );
        let mut t = Trainer::with_retina(Params::default(), train.retina(&s), 10, 7);
        let r = t.fit(&train, Some(&test), 1, &mut (), &Control::new(0));
        let acc = r[0].test_accuracy.unwrap();
        assert!(acc > 0.3, "test accuracy {acc}");
        assert!(r[0].activity.synops > r[0].activity.spikes);
        assert_eq!(r[0].confusion.as_ref().unwrap().total(), 200);
    }

    #[test]
    fn images_and_recordings_describe_themselves() {
        let data = synthetic(5, 1);
        let s = Saccades::default();
        assert_eq!(
            (data.picture(), data.retina(&s), data.steps(&s)),
            ((28, 28), 34, 300)
        );
        assert_eq!(data.source(&s).len(), 2);
        let mut ev = Events::new();
        let mut pic = Vec::new();
        data.load(0, &s, &mut ev, &mut pic);
        assert_eq!((pic.len(), ev.steps()), (784, 300));
    }
}
