// SPDX-License-Identifier: Apache-2.0

//! Helpers shared by `ftgsxr-desktop`'s integration tests that run the binary as a child
//! (R-16, §1.13, Q-105, Q-113): the child's validation layer writes its own log, chosen by
//! a settings file, which the test reads after the run.

use std::fs;
use std::path::{Path, PathBuf};

/// A Khronos validation layer settings file in `dir` that sends the layer's errors,
/// warnings (performance ones too) and information to `log`; the directory holding it, the
/// one `VK_LAYER_SETTINGS_PATH` names, is returned. The layer writes its log even beside
/// wgpu's own messenger.
pub(crate) fn layer_settings(dir: &Path, log: &Path) -> PathBuf {
    let settings = dir.join("layer-settings");
    fs::create_dir_all(&settings).unwrap();
    let _ = fs::remove_file(log);
    fs::write(
        settings.join("vk_layer_settings.txt"),
        format!(
            "khronos_validation.debug_action = VK_DBG_LAYER_ACTION_LOG_MSG\n\
             khronos_validation.report_flags = error,warn,perf,info\n\
             khronos_validation.log_filename = {}\n",
            log.display()
        ),
    )
    .unwrap();
    settings
}

/// The layer's log: whether it reported itself active (its status message at instance
/// creation) and the header line of each error or warning, the lines that read
/// `<message id>(ERROR / <type>): …` or `<message id>(WARNING / <type>): …`.
pub(crate) fn layer_log(log: &Path) -> (bool, Vec<String>) {
    let text = fs::read_to_string(log).unwrap_or_default();
    let active = text.contains("Khronos Validation Layer Active");
    let messages = text
        .lines()
        .filter(|line| {
            line.split_once('(').is_some_and(|(id, rest)| {
                !id.contains(' ')
                    && (rest.starts_with("ERROR / ") || rest.starts_with("WARNING / "))
            })
        })
        .map(str::to_owned)
        .collect();
    (active, messages)
}
