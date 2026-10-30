// SPDX-License-Identifier: Apache-2.0

//! Helpers the integration tests that set files aside share (U-29, U-30).

use std::path::Path;

/// The names of the files in `dir`, sorted.
pub fn names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .expect("list the test directory")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// Today's UTC date as `YYYY-MM-DD`, counted day by day from 1970-01-01 (an oracle
/// independent of the crate's date arithmetic).
pub fn utc_today() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("after 1970")
        .as_secs();
    let mut days = secs / 86_400;
    let mut year = 1970;
    loop {
        let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
        let len = if leap { 366 } else { 365 };
        if days < len {
            let months = [
                31,
                if leap { 29 } else { 28 },
                31,
                30,
                31,
                30,
                31,
                31,
                30,
                31,
                30,
                31,
            ];
            let mut month = 0;
            while days >= months[month] {
                days -= months[month];
                month += 1;
            }
            return format!("{year:04}-{:02}-{:02}", month + 1, days + 1);
        }
        days -= len;
        year += 1;
    }
}
