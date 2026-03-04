//! `--set name=value` overrides of a model's constants.

use std::str::FromStr;

/// Parses `val` as the type of the field `key` names. Floats must be finite.
pub fn value<T: FromStr>(key: &str, val: &str) -> Result<T, String> {
    let bad = || format!("{key}: bad value {val}");
    if val.parse::<f64>().is_ok_and(|v| !v.is_finite()) {
        return Err(bad());
    }
    val.parse().map_err(|_| bad())
}

pub fn unknown(key: &str) -> String {
    format!("unknown parameter {key}; --help lists them")
}

/// The `--set` flags given on a command line, in order.
#[derive(Debug, Clone, Default)]
pub struct Overrides(Vec<(String, String)>);

impl Overrides {
    /// Takes `--set name=value`; false for any other flag.
    pub fn flag(&mut self, flag: &str, val: &str) -> Result<bool, String> {
        if flag != "--set" {
            return Ok(false);
        }
        match val.split_once('=') {
            Some((k, v)) if !k.trim().is_empty() => {
                self.0.push((k.trim().into(), v.trim().into()));
                Ok(true)
            }
            _ => Err(format!("--set: expected name=value, got {val}")),
        }
    }

    /// Hands each override to `set`, which names the field and parses the value.
    pub fn apply<F>(&self, mut set: F) -> Result<(), String>
    where
        F: FnMut(&str, &str) -> Result<(), String>,
    {
        self.0.iter().try_for_each(|(k, v)| set(k, v))
    }

    /// Usage lines for `--set`, naming the constants a model accepts.
    pub fn usage(keys: &[&str]) -> String {
        const INDENT: &str = "                     ";
        let mut out = String::from("  --set NAME=VALUE   override a constant, repeatable; names:");
        let mut line = String::new();
        for k in keys {
            if !line.is_empty() && INDENT.len() + line.len() + k.len() + 2 > 78 {
                out += &format!("\n{INDENT}{}", line.trim_end());
                line.clear();
            }
            line += k;
            line += ", ";
        }
        out += &format!("\n{INDENT}{}", line.trim_end_matches(", "));
        out
    }
}

/// Makes a parameter struct settable by name: `KEYS` lists the names and `set`
/// parses a value into the field each one points at.
#[macro_export]
macro_rules! tunable {
    ($ty:ty { $($key:literal => $($field:ident).+),+ $(,)? }) => {
        impl $ty {
            pub const KEYS: &'static [&'static str] = &[$($key),+];

            pub fn set(&mut self, key: &str, val: &str) -> Result<(), String> {
                match key {
                    $($key => self.$($field).+ = $crate::tune::value(key, val)?,)+
                    _ => return Err($crate::tune::unknown(key)),
                }
                Ok(())
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Default, PartialEq)]
    struct Inner {
        rate: f32,
    }

    #[derive(Debug, Default, PartialEq)]
    struct Toy {
        steps: usize,
        inner: Inner,
    }

    crate::tunable!(Toy {
        "steps" => steps,
        "rate" => inner.rate,
    });

    #[test]
    fn values_parse_to_the_field_type() {
        assert_eq!(value::<usize>("n", "12"), Ok(12));
        assert_eq!(value::<f32>("x", "1e-3"), Ok(0.001));
        assert_eq!(value::<usize>("n", "1.5").unwrap_err(), "n: bad value 1.5");
        for v in ["nan", "inf", "-inf"] {
            assert!(value::<f32>("x", v).is_err(), "{v}");
        }
    }

    #[test]
    fn tunable_sets_nested_fields_by_name() {
        let mut t = Toy::default();
        t.set("steps", "30").unwrap();
        t.set("rate", "0.5").unwrap();
        assert_eq!(
            t,
            Toy {
                steps: 30,
                inner: Inner { rate: 0.5 }
            }
        );
        assert_eq!(Toy::KEYS, ["steps", "rate"]);
        assert!(t
            .set("speed", "1")
            .unwrap_err()
            .contains("unknown parameter speed"));
        assert!(t.set("steps", "-3").is_err());
    }

    #[test]
    fn overrides_apply_in_order_and_stop_at_the_first_error() {
        let mut o = Overrides::default();
        assert_eq!(o.flag("--seed", "1"), Ok(false));
        for v in ["steps=5", " rate = 2 ", "steps=9", "rate=x"] {
            assert_eq!(o.flag("--set", v), Ok(true));
        }
        let mut t = Toy::default();
        let err = o.apply(|k, v| t.set(k, v)).unwrap_err();
        assert_eq!(err, "rate: bad value x");
        assert_eq!(
            t,
            Toy {
                steps: 9,
                inner: Inner { rate: 2.0 }
            }
        );
        assert!(o.flag("--set", "steps").is_err());
        assert!(o.flag("--set", "=3").is_err());
    }

    #[test]
    fn usage_wraps_the_names() {
        let keys: Vec<String> = (0..30).map(|i| format!("param_{i}")).collect();
        let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
        let u = Overrides::usage(&keys);
        assert!(u.lines().all(|l| l.len() <= 80 && l == l.trim_end()), "{u}");
        assert!(u.lines().count() > 3 && u.contains("param_29") && !u.ends_with(','));
    }
}
