// SPDX-License-Identifier: GPL-3.0-or-later
// Host file paths are UTF-8, including those returned by SDL on Windows.
#ifndef GXRUNTIME_UTF8_FILE_H
#define GXRUNTIME_UTF8_FILE_H

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

// Returned strings are owned by the caller and released with free().
static inline wchar_t* dol_utf8_to_wide(const char* value) {
    int count;
    wchar_t* result;
    if (!value || !(count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value, -1, NULL, 0))) {
        errno = EINVAL;
        return NULL;
    }
    result = (wchar_t*)malloc((size_t)count * sizeof(wchar_t));
    if (!result) {
        errno = ENOMEM;
        return NULL;
    }
    if (!MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value, -1, result, count)) {
        free(result);
        errno = EINVAL;
        return NULL;
    }
    return result;
}

static inline char* dol_utf8_from_wide(const wchar_t* value) {
    int count;
    char* result;
    if (!value || !(count = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value, -1, NULL, 0, NULL, NULL))) {
        errno = EINVAL;
        return NULL;
    }
    result = (char*)malloc((size_t)count);
    if (!result) {
        errno = ENOMEM;
        return NULL;
    }
    if (!WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value, -1, result, count, NULL, NULL)) {
        free(result);
        errno = EINVAL;
        return NULL;
    }
    return result;
}
#endif

static inline FILE* dol_utf8_fopen(const char* path, const char* mode) {
#ifdef _WIN32
    wchar_t* wide_path = dol_utf8_to_wide(path);
    wchar_t* wide_mode;
    FILE* result;
    if (!wide_path)
        return NULL;
    wide_mode = dol_utf8_to_wide(mode);
    if (!wide_mode) {
        free(wide_path);
        return NULL;
    }
    result = _wfopen(wide_path, wide_mode);
    free(wide_mode);
    free(wide_path);
    return result;
#else
    return fopen(path, mode);
#endif
}

#ifdef __cplusplus
#include <string>
namespace dol_utf8 {
inline std::string environment(const char* name) {
#ifdef _WIN32
    wchar_t* wide_name = dol_utf8_to_wide(name);
    if (!wide_name)
        return {};
    const wchar_t* value = _wgetenv(wide_name);
    free(wide_name);
    if (!value)
        return {};
    char* utf8 = dol_utf8_from_wide(value);
    if (!utf8)
        return {};
    std::string result(utf8);
    free(utf8);
    return result;
#else
    const char* value = getenv(name);
    return value ? value : "";
#endif
}
} // namespace dol_utf8
#endif
#endif
