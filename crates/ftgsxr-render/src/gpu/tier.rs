// SPDX-License-Identifier: Apache-2.0

//! The quality tiers of §1.12 as one record each (D-37, D-12).

use ftgsxr_core::{GpuLayoutOptions, HalfFields};

/// E8's key width: the full float32 depth bits, or their top 16 bits (Q-2's `sort_key16`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortKey {
    /// 32-bit keys, four 8-bit radix passes (§1.4 stage 4).
    Depth32,
    /// 16-bit keys, two passes (§1.12, `quest-low`).
    Depth16,
}

/// How stereo views share the prepare stages (§1.4 Stereo, D-4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortMode {
    /// Stages 1–5 for each view with its own camera: exact.
    PerView,
    /// Stages 1–4 once at the eyes' midpoint, stage 5 per eye.
    SharedCenter,
}

/// The runtime's fixed-foveation level (§1.12 Headset display, D-27); `Off` where the
/// table has "—".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Foveation {
    /// No foveation.
    Off,
    /// The runtime's low level.
    Low,
    /// The runtime's medium level.
    Medium,
    /// The runtime's high level.
    High,
    /// The runtime's dynamic level.
    Dynamic,
}

/// The capacity C of a view's frame buffers (§1.4, §1.12, D-34).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Capacity {
    /// A fixed number of records.
    Fixed(u32),
    /// The loaded models' total N (the conformance tier).
    SumOfModels,
}

/// A quality tier (§1.12, D-37): every number that changes outputs or speed.
#[derive(Clone, Debug, PartialEq)]
pub struct QualityTier {
    /// The tier's name, as `by_name` takes it.
    pub name: String,
    /// Storage precision and the SH cap (§1.2 load options 1 and 2, D-12).
    pub layout: GpuLayoutOptions,
    /// E8's key width.
    pub sort_key: SortKey,
    /// Per-view or shared-center sorting (D-4).
    pub sort_mode: SortMode,
    /// The frame buffers' capacity C.
    pub capacity: Capacity,
    /// E13's environment occlusion.
    pub occlusion: bool,
    /// The swapchain's render scale.
    pub render_scale: f32,
    /// Fixed foveation.
    pub foveation: Foveation,
    /// Shared-center records with Σ^S in half precision (48 bytes, §1.4 stage 2).
    pub half_records: bool,
    /// The static model bytes a load may take (§1.12 Memory, D-13).
    pub memory_budget_bytes: u64,
}

const GIB: u64 = 1 << 30;

/// "1.5 GiB of model buffers on the Quest" (§1.12).
const QUEST_BUDGET: u64 = 3 * GIB / 2;

/// The desktop's budget when the adapter's memory cannot be queried (Q-66), and the
/// placeholder the desktop-class tiers hold until a device fills in 75 % of the adapter's
/// largest device-local heap.
pub(crate) const DESKTOP_FALLBACK_BUDGET: u64 = 8 * GIB; // Q-66

/// "velocity, scale, rotation, opacity, sh" (§1.12, the Quest rows); position, the
/// rebased time center and the duration stay float32.
const QUEST_HALF: HalfFields = HalfFields {
    velocity: true,
    scale: true,
    rotation: true,
    opacity: true,
    sh: true,
};

impl QualityTier {
    /// `conformance`: every approximation off (§1.12, §2.4 rule 3); `ftgsxr render`, R-n.
    pub fn conformance() -> Self {
        QualityTier {
            name: "conformance".into(),
            layout: GpuLayoutOptions {
                half: HalfFields::default(),
                sh_cap: None,
            },
            sort_key: SortKey::Depth32,
            sort_mode: SortMode::PerView,
            capacity: Capacity::SumOfModels,
            occlusion: false,
            render_scale: 1.0,
            foveation: Foveation::Off,
            half_records: false,
            memory_budget_bytes: DESKTOP_FALLBACK_BUDGET, // Q-66
        }
    }

    /// `desktop`: the default on the desktop (§1.12).
    pub fn desktop() -> Self {
        QualityTier {
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
            memory_budget_bytes: DESKTOP_FALLBACK_BUDGET, // Q-66
        }
    }

    /// `quest-high` (§1.12).
    pub fn quest_high() -> Self {
        Self::quest(
            "quest-high",
            3,
            SortKey::Depth32,
            1_500_000,
            1.0,
            Foveation::Low,
            false,
        )
    }

    /// `quest-balanced`: the default on the Quest (§1.12).
    pub fn quest_balanced() -> Self {
        Self::quest(
            "quest-balanced",
            2,
            SortKey::Depth32,
            1_000_000,
            1.0,
            Foveation::Medium,
            false,
        )
    }

    /// `quest-low` (§1.12), with half-precision shared-center records.
    pub fn quest_low() -> Self {
        Self::quest(
            "quest-low",
            1,
            SortKey::Depth16,
            600_000,
            0.8,
            Foveation::High,
            true,
        )
    }

    /// The tier `name` (`conformance`, `desktop`, `quest-high`, `quest-balanced`,
    /// `quest-low`), or `None`.
    pub fn by_name(name: &str) -> Option<Self> {
        match name {
            "conformance" => Some(Self::conformance()),
            "desktop" => Some(Self::desktop()),
            "quest-high" => Some(Self::quest_high()),
            "quest-balanced" => Some(Self::quest_balanced()),
            "quest-low" => Some(Self::quest_low()),
            _ => None,
        }
    }

    /// The three Quest rows, which differ only in these columns.
    fn quest(
        name: &str,
        sh_cap: u8,
        sort_key: SortKey,
        capacity: u32,
        render_scale: f32,
        foveation: Foveation,
        half_records: bool,
    ) -> Self {
        QualityTier {
            name: name.into(),
            layout: GpuLayoutOptions {
                half: QUEST_HALF,
                sh_cap: Some(sh_cap),
            },
            sort_key,
            sort_mode: SortMode::SharedCenter,
            capacity: Capacity::Fixed(capacity),
            occlusion: true,
            render_scale,
            foveation,
            half_records,
            memory_budget_bytes: QUEST_BUDGET,
        }
    }
}
