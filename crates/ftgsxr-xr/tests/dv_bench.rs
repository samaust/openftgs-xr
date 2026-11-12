// SPDX-License-Identifier: Apache-2.0

//! B-3, B-4, B-5 and B-6 on the headset (§3.5, `dv` lane, Q-406, Q-413): the driver reads the
//! headset's build fingerprint with `adb shell getprop ro.build.fingerprint`, launches the
//! app with the benchmark's name in the extra `ftgsxr-bench` and the fingerprint in
//! `ftgsxr-bench-headset`, waits for `ftgsxr-bench: <NAME> DONE|FAIL <detail>`, pulls
//! `dv/<NAME>.json` and checks that it is well formed (Q-67, Q-103) with
//! `machine.headset`, then prints its budgets. A P-3 or P-4 budget the Quest misses is
//! printed, not failed (spike S4); B-6's 1 % is the test's criterion. The runs leave
//! Android's GPU debug layers off, since they time the GPU (Q-418). Prerequisites:
//! `dv_support`'s.

use crate::dv_support;

use std::path::PathBuf;
use std::time::Duration;

use dv_support::{apk, bench_line, clear, device, getprop, install, launch_bench, pull, stop};
use serde_json::Value;

/// Runs the benchmark `name` on the headset and returns its pulled file, checked well
/// formed with the headset's fingerprint and the `quest-balanced` tier in every setup
/// but B-6's (Q-413).
fn run(name: &str, timeout: Duration) -> Value {
    let apk = apk();
    device();
    install(&apk);
    stop();
    clear();
    let fingerprint = getprop("ro.build.fingerprint");
    assert!(!fingerprint.is_empty(), "a build fingerprint");
    let start = launch_bench(name, &fingerprint);
    let line = bench_line(name, start, timeout);
    stop();
    let (done, detail) =
        line.unwrap_or_else(|| panic!("an ftgsxr-bench line for {name} (none: killed?)"));
    assert!(done, "{name} FAIL {detail}");
    println!(
        "{name} DONE {detail} in {:.0} s",
        start.elapsed().as_secs_f64()
    );
    let local: PathBuf =
        std::env::temp_dir().join(format!("ftgsxr-dv-{name}-{}.json", std::process::id()));
    pull(&format!("{name}.json"), &local);
    let text = std::fs::read_to_string(&local).expect("the pulled file");
    let _ = std::fs::remove_file(&local);
    println!("{text}");
    let file: Value = serde_json::from_str(&text).expect("well-formed JSON");
    // Q-67, Q-103's top level.
    assert_eq!(file["format"], "ftgsxr-bench");
    assert_eq!(file["version"], 1);
    assert_eq!(file["bench"], name);
    for field in ["gpu", "driver", "os", "cpu"] {
        assert!(file["machine"][field].is_string(), "machine.{field}");
    }
    assert!(file["machine"]["threads"].is_u64(), "machine.threads");
    // Q-406: the fingerprint, as given.
    assert_eq!(file["machine"]["headset"], fingerprint.as_str());
    for field in ["ftgsxr", "bevy", "wgpu"] {
        assert!(file["versions"][field].is_string(), "versions.{field}");
    }
    assert!(file["profile"].is_string());
    assert!(file["results"].is_array() && file["budgets"].is_array());
    let notes = file["notes"].as_array().expect("notes");
    // Q-116: the bench tests fail on GPU validation messages.
    assert!(
        !notes.iter().any(|n| n
            .as_str()
            .is_some_and(|n| n.contains("GPU validation reported"))),
        "{notes:?}"
    );
    for r in file["results"].as_array().expect("results") {
        // Q-413, Q-417: a number, or null with the reason the device refused the size.
        assert!(
            r["value"].is_number() || (r["value"].is_null() && r["skipped"].is_string()),
            "{r}"
        );
        assert!(r["measure"].is_string() && r["unit"].is_string(), "{r}");
    }
    file
}

/// The file's budgets, one line each, with the profile beside them (the dv APK's).
fn print_budgets(file: &Value) {
    let profile = file["profile"].as_str().unwrap_or("?");
    for b in file["budgets"].as_array().expect("budgets") {
        println!(
            "{} {} {}: {} {} {} -> {} ({profile} build)",
            b["id"].as_str().unwrap_or("?"),
            b["measure"].as_str().unwrap_or("?"),
            b["setup"],
            b["value"],
            b["rule"].as_str().unwrap_or("?"),
            b["limit"],
            if b["pass"] == true { "met" } else { "missed" }
        );
    }
    for r in file["results"].as_array().expect("results") {
        if let Some(reason) = r["skipped"].as_str() {
            println!("skipped {} {}: {reason}", r["measure"], r["setup"]);
        }
    }
}

/// The results of `measure`.
fn results<'a>(file: &'a Value, measure: &str) -> Vec<&'a Value> {
    file["results"]
        .as_array()
        .expect("results")
        .iter()
        .filter(|r| r["measure"] == measure)
        .collect()
}

#[test]
#[ignore = "dv"]
fn dv_b3_writes_the_sort_times_with_p4_on_the_quest() {
    // §3.5 B-3, Q-102's sizes, Q-413's tier; spike S4 reads the figures.
    let file = run("B-3", Duration::from_secs(20 * 60));
    print_budgets(&file);
    let sorts = results(&file, "sort_ms");
    let capacities: Vec<f64> = sorts
        .iter()
        .map(|r| r["setup"]["capacity"].as_f64().expect("a capacity"))
        .collect();
    assert_eq!(
        capacities,
        [1e5, 2.5e5, 5e5, 1e6, 2e6, 4e6],
        "one result for each C"
    );
    for r in &sorts {
        assert_eq!(r["setup"]["tier"], "quest-balanced", "{r}");
    }
    let budgets = file["budgets"].as_array().expect("budgets");
    assert_eq!(budgets.len(), 1, "{budgets:?}");
    let p4 = &budgets[0];
    assert_eq!(p4["id"], "P-4");
    assert_eq!(p4["setup"]["capacity"], 1_000_000);
    assert_eq!(p4["rule"], "at_most");
    // §1.12 P-4: ≤ 2.5 ms for C = 1 M on the Quest.
    assert_eq!(p4["limit"], 2.5);
}

#[test]
#[ignore = "dv"]
fn dv_b4_writes_the_cull_times_with_and_without_the_index_on_the_quest() {
    // §3.5 B-4 (Q-408, Q-T733.1): for each size, the cull time and the rows visited with
    // the temporal index and with every row, in Q-413's tier; a size the device refuses
    // is skipped in both setups. The B-4 checks are printed with the other budgets and
    // must hold: the index never visits more rows and is never slower beyond the tolerance (Q-T733.1).
    let file = run("B-4", Duration::from_secs(30 * 60));
    print_budgets(&file);
    for index in ["temporal", "every row"] {
        let sizes: Vec<f64> = results(&file, "cull_ms")
            .iter()
            .filter(|r| r["setup"]["index"] == index)
            .map(|r| {
                assert_eq!(r["setup"]["tier"], "quest-balanced", "{r}");
                r["setup"]["n"].as_f64().expect("n")
            })
            .collect();
        assert_eq!(
            sizes,
            [1e5, 2.5e5, 5e5, 1e6, 2e6, 3e6],
            "one cull_ms for each N, index {index}"
        );
    }
    for r in results(&file, "rows_visited") {
        if r["setup"]["index"] == "every row" && r["value"].is_number() {
            assert_eq!(r["value"], r["setup"]["n"], "{r}");
        }
    }
    let budgets = file["budgets"].as_array().expect("budgets");
    assert!(!budgets.is_empty(), "the B-4 checks of every size run");
    for b in budgets {
        assert_eq!(b["id"], "B-4", "{b}");
        assert_eq!(b["setup"]["index"], "temporal", "{b}");
        assert_eq!(b["pass"], true, "{b}");
    }
}

#[test]
#[ignore = "dv"]
fn dv_b5_writes_the_load_times_with_p3_on_the_quest() {
    // §3.5 B-5: files of 10, 100 and 500 MB, P-3 for each (Q-413's tier).
    let file = run("B-5", Duration::from_secs(40 * 60));
    print_budgets(&file);
    let budgets = file["budgets"].as_array().expect("budgets");
    let mbs: Vec<u64> = budgets
        .iter()
        .map(|b| {
            assert_eq!(b["id"], "P-3", "{b}");
            assert_eq!(b["measure"], "s_per_100mb", "{b}");
            assert_eq!(b["rule"], "at_most", "{b}");
            // §1.12 P-3: ≤ 1 s per 100 MB.
            assert_eq!(b["limit"], 1.0, "{b}");
            assert_eq!(b["setup"]["tier"], "quest-balanced", "{b}");
            let bytes = b["setup"]["file_bytes"].as_f64().expect("file_bytes");
            (bytes / 1e6).round() as u64
        })
        .collect();
    assert_eq!(mbs, [10, 100, 500], "P-3 for each file size");
}

/// §1.12 Memory: static GPU bytes per primitive at L′, float32 64 + 12 (L′+1)², half
/// precision 42 + 6 (L′+1)².
fn table(half: bool, l: u64) -> f64 {
    let k = ((l + 1) * (l + 1)) as f64;
    if half {
        42.0 + 6.0 * k
    } else {
        64.0 + 12.0 * k
    }
}

#[test]
#[ignore = "dv"]
fn dv_b6_matches_the_memory_table_within_1_percent_on_the_quest() {
    // §3.5 B-6: every tier and L against §1.12's table, within 1 %.
    let file = run("B-6", Duration::from_secs(10 * 60));
    print_budgets(&file);
    let budgets = file["budgets"].as_array().expect("budgets");
    assert_eq!(budgets.len(), 5 * 4, "every tier × L = 0 … 3");
    for b in budgets {
        let tier = b["setup"]["tier"].as_str().expect("a tier");
        let l = b["setup"]["sh_degree"].as_u64().expect("L");
        // §1.12's SH caps: the model's L, 3, 3, 2, 1; half precision on the Quest tiers.
        let (half, cap) = match tier {
            "conformance" => (false, 3),
            "desktop" => (false, 3),
            "quest-high" => (true, 3),
            "quest-balanced" => (true, 2),
            "quest-low" => (true, 1),
            other => panic!("tier {other}"),
        };
        let expected = table(half, l.min(cap));
        let value = b["value"].as_f64().expect("a value");
        assert!(
            (value / expected - 1.0).abs() <= 0.01,
            "{tier} L = {l}: {value} bytes against {expected}"
        );
        assert_eq!(b["pass"], true, "{b}");
    }
}
