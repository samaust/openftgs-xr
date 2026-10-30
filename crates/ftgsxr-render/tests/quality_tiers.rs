// SPDX-License-Identifier: Apache-2.0

//! The named quality tiers (`cpu` lane): each equals its row of §1.12's table (D-37,
//! D-12), with the memory budgets of §1.12's Memory paragraph. The desktop budget is
//! Q-66's placeholder (the 8 GiB fallback) until a device fills it in.

use ftgsxr_core::{GpuLayoutOptions, HalfFields};
use ftgsxr_render::{Capacity, Foveation, QualityTier, SortKey, SortMode};

const GIB: u64 = 1 << 30;
/// "1.5 GiB of model buffers on the Quest" (§1.12).
const QUEST_BUDGET: u64 = 3 * GIB / 2;
/// Q-66's fallback, held by the desktop-class tiers as a placeholder.
const DESKTOP_PLACEHOLDER: u64 = 8 * GIB;

/// "velocity, scale, rotation, opacity, sh" (§1.12, the Quest rows).
const QUEST_HALF: HalfFields = HalfFields {
    velocity: true,
    scale: true,
    rotation: true,
    opacity: true,
    sh: true,
};

fn expected(name: &str) -> QualityTier {
    match name {
        "conformance" => QualityTier {
            name: "conformance".into(),
            layout: GpuLayoutOptions {
                half: HalfFields::default(),
                sh_cap: None, // "the model's L"
            },
            sort_key: SortKey::Depth32,
            sort_mode: SortMode::PerView,
            capacity: Capacity::SumOfModels,
            occlusion: false,
            render_scale: 1.0,
            foveation: Foveation::Off, // "—"
            half_records: false,
            memory_budget_bytes: DESKTOP_PLACEHOLDER,
        },
        "desktop" => QualityTier {
            name: "desktop".into(),
            layout: GpuLayoutOptions {
                half: HalfFields::default(),
                sh_cap: Some(3),
            },
            sort_key: SortKey::Depth32,
            sort_mode: SortMode::PerView,
            capacity: Capacity::Fixed(4_000_000),
            occlusion: false,
            render_scale: 1.0,
            foveation: Foveation::Off,
            half_records: false,
            memory_budget_bytes: DESKTOP_PLACEHOLDER,
        },
        "quest-high" => QualityTier {
            name: "quest-high".into(),
            layout: GpuLayoutOptions {
                half: QUEST_HALF,
                sh_cap: Some(3),
            },
            sort_key: SortKey::Depth32,
            sort_mode: SortMode::SharedCenter,
            capacity: Capacity::Fixed(1_500_000),
            occlusion: true,
            render_scale: 1.0,
            foveation: Foveation::Low,
            half_records: false,
            memory_budget_bytes: QUEST_BUDGET,
        },
        "quest-balanced" => QualityTier {
            name: "quest-balanced".into(),
            layout: GpuLayoutOptions {
                half: QUEST_HALF,
                sh_cap: Some(2),
            },
            sort_key: SortKey::Depth32,
            sort_mode: SortMode::SharedCenter,
            capacity: Capacity::Fixed(1_000_000),
            occlusion: true,
            render_scale: 1.0,
            foveation: Foveation::Medium,
            half_records: false,
            memory_budget_bytes: QUEST_BUDGET,
        },
        "quest-low" => QualityTier {
            name: "quest-low".into(),
            layout: GpuLayoutOptions {
                half: QUEST_HALF,
                sh_cap: Some(1),
            },
            sort_key: SortKey::Depth16,
            sort_mode: SortMode::SharedCenter,
            capacity: Capacity::Fixed(600_000),
            occlusion: true,
            render_scale: 0.8,
            foveation: Foveation::High,
            half_records: true, // "quest-low also stores the shared-center records' Σ^S in half precision"
            memory_budget_bytes: QUEST_BUDGET,
        },
        other => panic!("no row {other}"),
    }
}

#[test]
fn quality_tiers_equal_their_rows_of_section_1_12() {
    let tiers = [
        QualityTier::conformance(),
        QualityTier::desktop(),
        QualityTier::quest_high(),
        QualityTier::quest_balanced(),
        QualityTier::quest_low(),
    ];
    let names = [
        "conformance",
        "desktop",
        "quest-high",
        "quest-balanced",
        "quest-low",
    ];
    for (tier, name) in tiers.into_iter().zip(names) {
        assert_eq!(tier, expected(name), "{name}");
    }
}

#[test]
fn quality_tier_by_name_finds_each_named_tier() {
    for name in [
        "conformance",
        "desktop",
        "quest-high",
        "quest-balanced",
        "quest-low",
    ] {
        assert_eq!(QualityTier::by_name(name), Some(expected(name)), "{name}");
    }
    assert_eq!(QualityTier::by_name("quest_low"), None);
    assert_eq!(QualityTier::by_name(""), None);
}
