//! Parsing and printing TIMESTAMP values, stored as microseconds since
//! 1970-01-01 00:00:00 UTC.

use std::fmt;

const MICROS_PER_SECOND: i64 = 1_000_000;
const SECONDS_PER_DAY: i64 = 86_400;
const MICROS_PER_DAY: i64 = MICROS_PER_SECOND * SECONDS_PER_DAY;
const FRACTION_DIGITS: usize = 6;

/// Days from 0000-03-01 to 1970-01-01 in the proleptic Gregorian calendar.
const DAYS_FROM_CIVIL_EPOCH_TO_UNIX_EPOCH: i64 = 719_468;
const DAYS_PER_400_YEARS: i64 = 146_097;

/// The text a TIMESTAMP literal was written with, when it is not a valid timestamp.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidTimestamp {
    pub literal: String,
}

impl fmt::Display for InvalidTimestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid TIMESTAMP '{}': expected YYYY-MM-DD[ HH:MM:SS[.ffffff]]",
            self.literal
        )
    }
}

impl std::error::Error for InvalidTimestamp {}

/// Parses `YYYY-MM-DD`, `YYYY-MM-DD HH:MM:SS` or `YYYY-MM-DDTHH:MM:SS`, with up to six
/// fractional second digits, as a UTC time. Returns microseconds since the Unix epoch.
pub fn parse(literal: &str) -> Result<i64, InvalidTimestamp> {
    parse_parts(literal.trim()).ok_or_else(|| InvalidTimestamp {
        literal: literal.to_string(),
    })
}

fn parse_parts(s: &str) -> Option<i64> {
    let (date, time) = match s.split_once([' ', 'T']) {
        Some((date, time)) => (date, Some(time.trim_start())),
        None => (s, None),
    };
    let days = parse_date(date)?;
    let micros_of_day = time.map_or(Some(0), parse_time)?;
    days.checked_mul(MICROS_PER_DAY)?.checked_add(micros_of_day)
}

fn parse_date(date: &str) -> Option<i64> {
    let mut parts = date.split('-');
    let year = fixed_digits(parts.next()?, 4)?;
    let month = fixed_digits(parts.next()?, 2)?;
    let day = fixed_digits(parts.next()?, 2)?;
    if parts.next().is_some() || !(1..=12).contains(&month) {
        return None;
    }
    if day < 1 || day > days_in_month(year, month) {
        return None;
    }
    Some(days_from_civil(year, month, day))
}

fn parse_time(time: &str) -> Option<i64> {
    let (hms, fraction) = match time.split_once('.') {
        Some((hms, fraction)) => (hms, Some(fraction)),
        None => (time, None),
    };
    let mut parts = hms.split(':');
    let hour = fixed_digits(parts.next()?, 2)?;
    let minute = fixed_digits(parts.next()?, 2)?;
    let second = fixed_digits(parts.next()?, 2)?;
    if parts.next().is_some() || hour > 23 || minute > 59 || second > 59 {
        return None;
    }
    let micros = match fraction {
        None => 0,
        Some(digits) => {
            if digits.is_empty()
                || digits.len() > FRACTION_DIGITS
                || !digits.bytes().all(|b| b.is_ascii_digit())
            {
                return None;
            }
            let padded = format!("{digits:0<FRACTION_DIGITS$}");
            padded.parse::<i64>().ok()?
        }
    };
    Some(((hour * 60 + minute) * 60 + second) * MICROS_PER_SECOND + micros)
}

/// Parses exactly `len` ASCII digits.
fn fixed_digits(s: &str, len: usize) -> Option<i64> {
    (s.len() == len && s.bytes().all(|b| b.is_ascii_digit()))
        .then(|| s.parse().ok())
        .flatten()
}

fn is_leap_year(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        2 if is_leap_year(year) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Days since 1970-01-01 of a proleptic Gregorian date (Howard Hinnant's `days_from_civil`).
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let year_of_era = year - era * 400;
    let month_from_march = (month + 9) % 12;
    let day_of_year = (153 * month_from_march + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * DAYS_PER_400_YEARS + day_of_era - DAYS_FROM_CIVIL_EPOCH_TO_UNIX_EPOCH
}

/// The proleptic Gregorian `(year, month, day)` of a day count since 1970-01-01.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let days = days + DAYS_FROM_CIVIL_EPOCH_TO_UNIX_EPOCH;
    let era = days.div_euclid(DAYS_PER_400_YEARS);
    let day_of_era = days - era * DAYS_PER_400_YEARS;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_from_march = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_from_march + 2) / 5 + 1;
    let month = if month_from_march < 10 {
        month_from_march + 3
    } else {
        month_from_march - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

/// Prints microseconds since the Unix epoch as `YYYY-MM-DD HH:MM:SS`, followed by the
/// fraction of a second when it is not zero.
pub fn format(micros: i64) -> String {
    let days = micros.div_euclid(MICROS_PER_DAY);
    let micros_of_day = micros.rem_euclid(MICROS_PER_DAY);
    let (year, month, day) = civil_from_days(days);
    let seconds_of_day = micros_of_day / MICROS_PER_SECOND;
    let fraction = micros_of_day % MICROS_PER_SECOND;
    let (hour, minute, second) = (
        seconds_of_day / 3600,
        seconds_of_day / 60 % 60,
        seconds_of_day % 60,
    );
    let mut out = format!("{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02}");
    if fraction != 0 {
        let digits = format!("{fraction:06}");
        out.push('.');
        out.push_str(digits.trim_end_matches('0'));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_dates_and_times_as_utc_microseconds() {
        assert_eq!(parse("1970-01-01"), Ok(0));
        assert_eq!(parse("1970-01-01 00:00:01"), Ok(MICROS_PER_SECOND));
        assert_eq!(
            parse("2023-11-14 22:13:20"),
            Ok(1_700_000_000 * MICROS_PER_SECOND)
        );
        assert_eq!(
            parse("2023-11-14T22:13:20"),
            Ok(1_700_000_000 * MICROS_PER_SECOND)
        );
        assert_eq!(parse("1970-01-01 00:00:00.5"), Ok(500_000));
        assert_eq!(parse("1970-01-01 00:00:00.000001"), Ok(1));
        assert_eq!(parse("1969-12-31 23:59:59"), Ok(-MICROS_PER_SECOND));
        assert_eq!(parse("  2024-02-29  "), parse("2024-02-29"));
    }

    #[test]
    fn rejects_malformed_and_impossible_timestamps() {
        for bad in [
            "",
            "2024",
            "2024-1-01",
            "2024-13-01",
            "2024-00-10",
            "2023-02-29",
            "2024-04-31",
            "2024-01-01 24:00:00",
            "2024-01-01 12:60:00",
            "2024-01-01 12:00",
            "2024-01-01 12:00:00.",
            "2024-01-01 12:00:00.1234567",
            "2024-01-01 12:00:00+02",
            "yesterday",
        ] {
            assert!(parse(bad).is_err(), "{bad:?} is not a timestamp");
        }
    }

    #[test]
    fn format_round_trips_through_parse() {
        for text in [
            "1970-01-01 00:00:00",
            "2000-02-29 12:34:56",
            "1969-07-20 20:17:40",
            "0001-01-01 00:00:00",
            "9999-12-31 23:59:59.999999",
            "2024-01-01 00:00:00.25",
        ] {
            let micros = parse(text).unwrap();
            assert_eq!(format(micros), text, "{text} prints as it was written");
        }
    }

    #[test]
    fn format_prints_every_i64() {
        for micros in [i64::MIN, -1, 0, 1, i64::MAX] {
            assert!(!format(micros).is_empty());
        }
    }

    #[test]
    fn invalid_timestamp_names_the_literal_and_the_format() {
        let err = parse("2024-13-01").unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid TIMESTAMP '2024-13-01': expected YYYY-MM-DD[ HH:MM:SS[.ffffff]]"
        );
    }
}
