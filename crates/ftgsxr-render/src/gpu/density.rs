// SPDX-License-Identifier: Apache-2.0

//! The XR device's VK\_EXT\_fragment\_density\_map features and properties, read through
//! `wgpu::hal` (§2.6: `unsafe` in `ftgsxr-render` for wgpu-hal interop; D-58,
//! Appendix E): what spike S-F1 logs before its measurements (#805, Q-454) and the
//! support signal the layer pass will read.
// Q-T805.1

use std::ffi::{CStr, c_void};
use std::fmt;

/// `VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2` and `…_PROPERTIES_2` (Vulkan 1.1), and
/// the extension's `…_FRAGMENT_DENSITY_MAP_FEATURES_EXT` and `…_PROPERTIES_EXT`, the
/// values of ash 0.38 (MIT OR Apache-2.0, source-log row T805.1).
const FEATURES_2: i32 = 1_000_059_000;
const PROPERTIES_2: i32 = 1_000_059_001;
const FDM_FEATURES: i32 = 1_000_218_000;
const FDM_PROPERTIES: i32 = 1_000_218_001;

const FDM: &CStr = c"VK_EXT_fragment_density_map";
const FDM2: &CStr = c"VK_EXT_fragment_density_map2";
const OFFSET_QCOM: &CStr = c"VK_QCOM_fragment_density_map_offset";
const OFFSET_EXT: &CStr = c"VK_EXT_fragment_density_map_offset";

/// `VkPhysicalDeviceFeatures2`: 55 `VkBool32` features after the header.
#[repr(C)]
struct Features2 {
    s_type: i32,
    p_next: *mut c_void,
    features: [u32; 55],
}

/// `VkPhysicalDeviceFragmentDensityMapFeaturesEXT`.
#[repr(C)]
struct FdmFeatures {
    s_type: i32,
    p_next: *mut c_void,
    map: u32,
    dynamic: u32,
    non_subsampled: u32,
}

/// `VkPhysicalDeviceProperties2`: the header, then `VkPhysicalDeviceProperties`, whose
/// size is taken from ash's own type (below) so no layout is restated.
#[repr(C)]
struct Properties2<P> {
    s_type: i32,
    p_next: *mut c_void,
    properties: P,
}

/// `VkPhysicalDeviceFragmentDensityMapPropertiesEXT`.
#[repr(C)]
struct FdmProperties {
    s_type: i32,
    p_next: *mut c_void,
    min: [u32; 2],
    max: [u32; 2],
    invocations: u32,
}

/// The density-map support of a Vulkan adapter and its device (Q-T805.1).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DensityMapSupport {
    /// The adapter offers VK\_EXT\_fragment\_density\_map.
    pub extension: bool,
    /// The extension is enabled on the device.
    pub enabled: bool,
    /// `fragmentDensityMap`.
    pub map: bool,
    /// `fragmentDensityMapDynamic`.
    pub dynamic: bool,
    /// `fragmentDensityMapNonSubsampledImages`.
    pub non_subsampled: bool,
    /// `minFragmentDensityTexelSize`, pixels (0 without the extension).
    pub min_texel: [u32; 2],
    /// `maxFragmentDensityTexelSize`, pixels (0 without the extension).
    pub max_texel: [u32; 2],
    /// `fragmentDensityInvocations`.
    pub invocations: bool,
    /// VK\_EXT\_fragment\_density\_map2 is offered.
    pub map2: bool,
    /// VK\_QCOM\_fragment\_density\_map\_offset is offered.
    pub offset_qcom: bool,
    /// VK\_EXT\_fragment\_density\_map\_offset is offered.
    pub offset_ext: bool,
}

impl fmt::Display for DensityMapSupport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VK_EXT_fragment_density_map {} (enabled {}), fragmentDensityMap {}, \
             fragmentDensityMapDynamic {}, fragmentDensityMapNonSubsampledImages {}, \
             minFragmentDensityTexelSize {}x{}, maxFragmentDensityTexelSize {}x{}, \
             fragmentDensityInvocations {}, VK_EXT_fragment_density_map2 {}, \
             VK_QCOM_fragment_density_map_offset {}, VK_EXT_fragment_density_map_offset {}",
            self.extension,
            self.enabled,
            self.map,
            self.dynamic,
            self.non_subsampled,
            self.min_texel[0],
            self.min_texel[1],
            self.max_texel[0],
            self.max_texel[1],
            self.invocations,
            self.map2,
            self.offset_qcom,
            self.offset_ext
        )
    }
}

/// The density-map support of `adapter` and `device` (`device` made from `adapter`);
/// `None` when they are not Vulkan (Q-T805.1).
#[expect(
    clippy::missing_transmute_annotations,
    reason = "ash's types cannot be named without depending on ash (§2.3)"
)]
pub fn density_map_support(
    adapter: &wgpu::Adapter,
    device: &wgpu::Device,
) -> Option<DensityMapSupport> {
    // SAFETY: the hal device's guard only reads the list of extensions the device was
    // created with, and is dropped at the block's end.
    let enabled = unsafe {
        let hal = device.as_hal::<wgpu::hal::api::Vulkan>()?;
        hal.enabled_device_extensions().contains(&FDM)
    };
    let mut s = DensityMapSupport {
        enabled,
        ..Default::default()
    };
    // SAFETY: as `largest_device_local_heap`: the adapter's guard is held for this block
    // only, nothing is destroyed or mutated, and the queries only read properties of the
    // physical device the guard's instance enumerated. The two `*2` entry points are core
    // Vulkan 1.1, which wgpu-hal requires of the instance it uses; each is called through
    // its own pointer transmuted to a signature over `#[repr(C)]` structs of the
    // specification's layouts (the handle is ash's `#[repr(transparent)]` pointer), whose
    // `pNext` chains end in null and outlive the calls.
    unsafe {
        let hal = adapter.as_hal::<wgpu::hal::api::Vulkan>()?;
        let instance = hal.shared_instance().raw_instance();
        let pd = hal.raw_physical_device();
        let exts = instance
            .enumerate_device_extension_properties(pd)
            .unwrap_or_default();
        let has = |name: &CStr| {
            exts.iter()
                .any(|e| e.extension_name_as_c_str().is_ok_and(|n| n == name))
        };
        s.extension = has(FDM);
        s.map2 = has(FDM2);
        s.offset_qcom = has(OFFSET_QCOM);
        s.offset_ext = has(OFFSET_EXT);
        if s.extension {
            let fp = instance.fp_v1_1();
            let mut fdm = FdmFeatures {
                s_type: FDM_FEATURES,
                p_next: std::ptr::null_mut(),
                map: 0,
                dynamic: 0,
                non_subsampled: 0,
            };
            let mut f2 = Features2 {
                s_type: FEATURES_2,
                p_next: (&raw mut fdm).cast(),
                features: [0; 55],
            };
            let get_features: unsafe extern "system" fn(*mut c_void, *mut Features2) =
                std::mem::transmute(fp.get_physical_device_features2);
            get_features(std::mem::transmute::<_, *mut c_void>(pd), &raw mut f2);
            s.map = fdm.map != 0;
            s.dynamic = fdm.dynamic != 0;
            s.non_subsampled = fdm.non_subsampled != 0;
            let mut props = FdmProperties {
                s_type: FDM_PROPERTIES,
                p_next: std::ptr::null_mut(),
                min: [0; 2],
                max: [0; 2],
                invocations: 0,
            };
            // The core properties' size, from ash's own value of them.
            let core = instance.get_physical_device_properties(pd);
            let mut p2 = Properties2 {
                s_type: PROPERTIES_2,
                p_next: (&raw mut props).cast(),
                properties: core,
            };
            let get_properties: unsafe extern "system" fn(*mut c_void, *mut Properties2<_>) =
                std::mem::transmute(fp.get_physical_device_properties2);
            get_properties(std::mem::transmute::<_, *mut c_void>(pd), &raw mut p2);
            s.min_texel = props.min;
            s.max_texel = props.max;
            s.invocations = props.invocations != 0;
        }
    }
    Some(s)
}
