// SPDX-License-Identifier: Apache-2.0

//! The JNI adapter of the shim (§2.5 The shim): the static methods of
//! `io.github.samaust.ftgsxr.Shim`, called from any thread (the JavaVM attaches it), the
//! class found once through the activity's class loader, since a natively attached
//! thread's `FindClass` sees only the system classes. A Java exception becomes `Err` with
//! its text, and is cleared.

use std::sync::OnceLock;

use jni::objects::{GlobalRef, JObject, JString, JValue};
use jni::{JNIEnv, JavaVM};

/// The shim's class name, as `ClassLoader.loadClass` takes it.
const SHIM: &str = "io.github.samaust.ftgsxr.Shim";

/// The shim's class, once loaded.
static CLASS: OnceLock<GlobalRef> = OnceLock::new();

/// Runs `f` with this thread's JNI environment and the shim's class; a JNI error or a Java
/// exception is `Err` with its text.
fn with_shim<T>(
    f: impl FnOnce(&JNIEnv<'_>, &GlobalRef) -> jni::errors::Result<T>,
) -> Result<T, String> {
    let ctx = ndk_context::android_context();
    // SAFETY: `ndk_context` holds the process's JavaVM pointer, set by android-activity
    // before `android_main` runs and valid for the process's lifetime.
    let vm = unsafe { JavaVM::from_raw(ctx.vm().cast()) }.map_err(|e| e.to_string())?;
    let env = vm.attach_current_thread().map_err(|e| e.to_string())?;
    let result = class(&env, ctx.context()).and_then(|class| f(&env, class));
    result.map_err(|e| match e {
        jni::errors::Error::JavaException => exception_text(&env),
        other => other.to_string(),
    })
}

/// The shim's class, loaded through the activity's class loader on first use.
fn class<'a>(
    env: &JNIEnv<'_>,
    context: *mut std::ffi::c_void,
) -> jni::errors::Result<&'a GlobalRef> {
    if let Some(class) = CLASS.get() {
        return Ok(class);
    }
    // SAFETY: `context` is the activity's global reference, which android-activity keeps
    // alive while the native code runs; the local `JObject` does not outlive this call.
    let activity = unsafe { JObject::from_raw(context.cast()) };
    let loader = env
        .call_method(activity, "getClassLoader", "()Ljava/lang/ClassLoader;", &[])?
        .l()?;
    let name = env.new_string(SHIM)?;
    let class = env
        .call_method(
            loader,
            "loadClass",
            "(Ljava/lang/String;)Ljava/lang/Class;",
            &[JValue::Object(name.into())],
        )?
        .l()?;
    let global = env.new_global_ref(class)?;
    Ok(CLASS.get_or_init(|| global))
}

/// The pending exception's `toString()`, cleared; or a generic text when it cannot be read.
fn exception_text(env: &JNIEnv<'_>) -> String {
    let Ok(throwable) = env.exception_occurred() else {
        return "a Java exception".into();
    };
    let _ = env.exception_clear();
    env.call_method(throwable, "toString", "()Ljava/lang/String;", &[])
        .and_then(|v| v.l())
        .and_then(|s| string(env, s))
        .ok()
        .flatten()
        .unwrap_or_else(|| {
            let _ = env.exception_clear();
            "a Java exception".into()
        })
}

/// A Java string, or `None` for null.
fn string(env: &JNIEnv<'_>, obj: JObject<'_>) -> jni::errors::Result<Option<String>> {
    if obj.is_null() {
        return Ok(None);
    }
    Ok(Some(env.get_string(JString::from(obj))?.into()))
}

/// A Java `String[]` as strings (a null element skipped); `None` for null.
fn strings(env: &JNIEnv<'_>, array: JObject<'_>) -> jni::errors::Result<Option<Vec<String>>> {
    if array.is_null() {
        return Ok(None);
    }
    let raw = array.into_raw();
    let n = env.get_array_length(raw)?;
    let mut out = Vec::with_capacity(n.max(0) as usize);
    for i in 0..n {
        let element = env.get_object_array_element(raw, i)?;
        if let Some(s) = string(env, element)? {
            out.push(s);
        }
        env.delete_local_ref(element)?;
    }
    Ok(Some(out))
}

/// `Shim.pickDocuments(requestId, tree)`.
pub(crate) fn pick_documents(request_id: i32, tree: bool) -> Result<(), String> {
    with_shim(|env, class| {
        env.call_static_method(
            class,
            "pickDocuments",
            "(IZ)V",
            &[JValue::Int(request_id), JValue::Bool(u8::from(tree))],
        )?;
        Ok(())
    })
}

/// `Shim.pollPick(requestId)`: the content URIs, `None` while pending, empty when cancelled.
pub(crate) fn poll_pick(request_id: i32) -> Result<Option<Vec<String>>, String> {
    with_shim(|env, class| {
        let array = env
            .call_static_method(
                class,
                "pollPick",
                "(I)[Ljava/lang/String;",
                &[JValue::Int(request_id)],
            )?
            .l()?;
        strings(env, array)
    })
}

/// `Shim.listTree(treeUri)`: one `name\tsize\tmime\turi` line per document.
pub(crate) fn list_tree(tree_uri: &str) -> Result<Vec<String>, String> {
    with_shim(|env, class| {
        let uri = env.new_string(tree_uri)?;
        let array = env
            .call_static_method(
                class,
                "listTree",
                "(Ljava/lang/String;)[Ljava/lang/String;",
                &[JValue::Object(uri.into())],
            )?
            .l()?;
        Ok(strings(env, array)?.unwrap_or_default())
    })
}

/// `Shim.openDocument(uri)`: a detached read-only file descriptor, −1 on failure.
pub(crate) fn open_document(uri: &str) -> Result<i32, String> {
    with_shim(|env, class| {
        let uri = env.new_string(uri)?;
        env.call_static_method(
            class,
            "openDocument",
            "(Ljava/lang/String;)I",
            &[JValue::Object(uri.into())],
        )?
        .i()
    })
}

/// `Shim.requestPermission(requestId, permission)`.
pub(crate) fn request_permission(request_id: i32, permission: &str) -> Result<(), String> {
    with_shim(|env, class| {
        let permission = env.new_string(permission)?;
        env.call_static_method(
            class,
            "requestPermission",
            "(ILjava/lang/String;)V",
            &[JValue::Int(request_id), JValue::Object(permission.into())],
        )?;
        Ok(())
    })
}

/// `Shim.pollPermission(requestId)`: −1 pending, 0 denied, 1 granted.
pub(crate) fn poll_permission(request_id: i32) -> Result<i32, String> {
    with_shim(|env, class| {
        env.call_static_method(class, "pollPermission", "(I)I", &[JValue::Int(request_id)])?
            .i()
    })
}

/// `Shim.startCamera()`: 0 opening, 1 no passthrough camera, 2 a camera permission
/// missing, 3 the open failed (Q-386).
// Q-386
pub(crate) fn start_camera() -> Result<i32, String> {
    with_shim(|env, class| {
        env.call_static_method(class, "startCamera", "()I", &[])?
            .i()
    })
}

/// `Shim.pollCameraThumb()`: the newest thumbnail's array once, `None` until the next.
// Q-386
pub(crate) fn poll_camera_thumb() -> Result<Option<Vec<f32>>, String> {
    with_shim(|env, class| {
        let array = env
            .call_static_method(class, "pollCameraThumb", "()[F", &[])?
            .l()?;
        if array.is_null() {
            return Ok(None);
        }
        let raw = array.into_raw();
        let n = env.get_array_length(raw)?;
        let mut out = vec![0.0f32; n.max(0) as usize];
        env.get_float_array_region(raw, 0, &mut out)?;
        env.delete_local_ref(array)?;
        Ok(Some(out))
    })
}

/// `Shim.stopCamera()`.
pub(crate) fn stop_camera() -> Result<(), String> {
    with_shim(|env, class| {
        env.call_static_method(class, "stopCamera", "()V", &[])?;
        Ok(())
    })
}
