// SPDX-License-Identifier: Apache-2.0

//! The activity's methods android-activity does not wrap, through JNI (§2.5): the
//! launch intent's string extra (the device tests, §3.4) and the cache directory (Q-15).

use std::path::PathBuf;

use jni::JavaVM;
use jni::objects::{JObject, JString, JValue};

/// Runs `f` with the JNI environment of this thread and the activity object.
fn with_activity<T>(
    f: impl FnOnce(&jni::JNIEnv<'_>, JObject<'_>) -> jni::errors::Result<T>,
) -> Option<T> {
    let ctx = ndk_context::android_context();
    // SAFETY: `ndk_context` holds the process's JavaVM pointer, set by android-activity
    // before `android_main` runs and valid for the process's lifetime.
    let vm = unsafe { JavaVM::from_raw(ctx.vm().cast()) }.ok()?;
    let env = vm.attach_current_thread().ok()?;
    // SAFETY: `context()` is the activity's global reference, which android-activity keeps
    // alive while the native code runs; the local `JObject` does not outlive this call.
    let activity = unsafe { JObject::from_raw(ctx.context().cast()) };
    let result = f(&env, activity);
    if env.exception_check().unwrap_or(false) {
        let _ = env.exception_clear();
    }
    result.ok()
}

/// A Java string, or `None` for null.
fn string(env: &jni::JNIEnv<'_>, obj: JObject<'_>) -> jni::errors::Result<Option<String>> {
    if obj.is_null() {
        return Ok(None);
    }
    Ok(Some(env.get_string(JString::from(obj))?.into()))
}

/// The launch intent's string extra `name` (`getIntent().getStringExtra(name)`).
pub(crate) fn intent_extra(name: &str) -> Option<String> {
    with_activity(|env, activity| {
        let intent = env
            .call_method(activity, "getIntent", "()Landroid/content/Intent;", &[])?
            .l()?;
        if intent.is_null() {
            return Ok(None);
        }
        let key = env.new_string(name)?;
        let value = env
            .call_method(
                intent,
                "getStringExtra",
                "(Ljava/lang/String;)Ljava/lang/String;",
                &[JValue::Object(key.into())],
            )?
            .l()?;
        string(env, value)
    })
    .flatten()
}

/// The app's cache directory (`getCacheDir().getAbsolutePath()`), which android-activity
/// 0.6.0 has no accessor for (Q-15, §2.5 Paths and storage).
pub(crate) fn cache_dir() -> Option<PathBuf> {
    with_activity(|env, activity| {
        let file = env
            .call_method(activity, "getCacheDir", "()Ljava/io/File;", &[])?
            .l()?;
        if file.is_null() {
            return Ok(None);
        }
        let path = env
            .call_method(file, "getAbsolutePath", "()Ljava/lang/String;", &[])?
            .l()?;
        string(env, path)
    })
    .flatten()
    .map(PathBuf::from)
}
