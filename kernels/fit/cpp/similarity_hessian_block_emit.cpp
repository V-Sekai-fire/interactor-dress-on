#ifndef SLANG_CPP_PRELUDE_H
#define SLANG_CPP_PRELUDE_H

// Because the signature of isnan, isfinite, and is isinf changed in C++, we use the macro
// to use the version in the std namespace.
// https://stackoverflow.com/questions/39130040/cmath-hides-isnan-in-math-h-in-c14-c11

#ifdef SLANG_LLVM
#ifndef SLANG_LLVM_H
#define SLANG_LLVM_H

// TODO(JS):
// Disable exception declspecs, as not supported on LLVM without some extra options.
// We could enable with `-fms-extensions`
#define SLANG_DISABLE_EXCEPTIONS 1

#ifndef SLANG_PRELUDE_ASSERT
#ifdef SLANG_PRELUDE_ENABLE_ASSERT
extern "C" void assertFailure(const char* msg);
#define SLANG_PRELUDE_EXPECT(VALUE, MSG) \
    if (VALUE)                           \
    {                                    \
    }                                    \
    else                                 \
        assertFailure("assertion failed: '" MSG "'")
#define SLANG_PRELUDE_ASSERT(VALUE) SLANG_PRELUDE_EXPECT(VALUE, #VALUE)
#else // SLANG_PRELUDE_ENABLE_ASSERT
#define SLANG_PRELUDE_EXPECT(VALUE, MSG)
#define SLANG_PRELUDE_ASSERT(x)
#endif // SLANG_PRELUDE_ENABLE_ASSERT
#endif

/*
Taken from stddef.h
*/

typedef __PTRDIFF_TYPE__ ptrdiff_t;
typedef __SIZE_TYPE__ size_t;
typedef __SIZE_TYPE__ rsize_t;

// typedef __WCHAR_TYPE__ wchar_t;

#if defined(__need_NULL)
#undef NULL
#ifdef __cplusplus
#if !defined(__MINGW32__) && !defined(_MSC_VER)
#define NULL __null
#else
#define NULL 0
#endif
#else
#define NULL ((void*)0)
#endif
#ifdef __cplusplus
#if defined(_MSC_EXTENSIONS) && defined(_NATIVE_NULLPTR_SUPPORTED)
namespace std
{
typedef decltype(nullptr) nullptr_t;
}
using ::std::nullptr_t;
#endif
#endif
#undef __need_NULL
#endif /* defined(__need_NULL) */


/*
The following are taken verbatim from stdint.h from Clang in LLVM. Only 8/16/32/64 types are needed.
*/

// LLVM/Clang types such that we can use LLVM/Clang without headers for C++ output from Slang

#ifdef __INT64_TYPE__
#ifndef __int8_t_defined /* glibc sys/types.h also defines int64_t*/
typedef __INT64_TYPE__ int64_t;
#endif /* __int8_t_defined */
typedef __UINT64_TYPE__ uint64_t;
#define __int_least64_t int64_t
#define __uint_least64_t uint64_t
#endif /* __INT64_TYPE__ */

#ifdef __int_least64_t
typedef __int_least64_t int_least64_t;
typedef __uint_least64_t uint_least64_t;
typedef __int_least64_t int_fast64_t;
typedef __uint_least64_t uint_fast64_t;
#endif /* __int_least64_t */

#ifdef __INT32_TYPE__

#ifndef __int8_t_defined /* glibc sys/types.h also defines int32_t*/
typedef __INT32_TYPE__ int32_t;
#endif /* __int8_t_defined */

#ifndef __uint32_t_defined /* more glibc compatibility */
#define __uint32_t_defined
typedef __UINT32_TYPE__ uint32_t;
#endif /* __uint32_t_defined */

#define __int_least32_t int32_t
#define __uint_least32_t uint32_t
#endif /* __INT32_TYPE__ */

#ifdef __int_least32_t
typedef __int_least32_t int_least32_t;
typedef __uint_least32_t uint_least32_t;
typedef __int_least32_t int_fast32_t;
typedef __uint_least32_t uint_fast32_t;
#endif /* __int_least32_t */

#ifdef __INT16_TYPE__
#ifndef __int8_t_defined /* glibc sys/types.h also defines int16_t*/
typedef __INT16_TYPE__ int16_t;
#endif /* __int8_t_defined */
typedef __UINT16_TYPE__ uint16_t;
#define __int_least16_t int16_t
#define __uint_least16_t uint16_t
#endif /* __INT16_TYPE__ */

#ifdef __int_least16_t
typedef __int_least16_t int_least16_t;
typedef __uint_least16_t uint_least16_t;
typedef __int_least16_t int_fast16_t;
typedef __uint_least16_t uint_fast16_t;
#endif /* __int_least16_t */

#ifdef __INT8_TYPE__
#ifndef __int8_t_defined /* glibc sys/types.h also defines int8_t*/
typedef __INT8_TYPE__ int8_t;
#endif /* __int8_t_defined */
typedef __UINT8_TYPE__ uint8_t;
#define __int_least8_t int8_t
#define __uint_least8_t uint8_t
#endif /* __INT8_TYPE__ */

#ifdef __int_least8_t
typedef __int_least8_t int_least8_t;
typedef __uint_least8_t uint_least8_t;
typedef __int_least8_t int_fast8_t;
typedef __uint_least8_t uint_fast8_t;
#endif /* __int_least8_t */

/* prevent glibc sys/types.h from defining conflicting types */
#ifndef __int8_t_defined
#define __int8_t_defined
#endif /* __int8_t_defined */

/* C99 7.18.1.4 Integer types capable of holding object pointers.
 */
#define __stdint_join3(a, b, c) a##b##c

#ifndef _INTPTR_T
#ifndef __intptr_t_defined
typedef __INTPTR_TYPE__ intptr_t;
#define __intptr_t_defined
#define _INTPTR_T
#endif
#endif

#ifndef _UINTPTR_T
typedef __UINTPTR_TYPE__ uintptr_t;
#define _UINTPTR_T
#endif

/* C99 7.18.1.5 Greatest-width integer types.
 */
typedef __INTMAX_TYPE__ intmax_t;
typedef __UINTMAX_TYPE__ uintmax_t;

/* C99 7.18.4 Macros for minimum-width integer constants.
 *
 * The standard requires that integer constant macros be defined for all the
 * minimum-width types defined above. As 8-, 16-, 32-, and 64-bit minimum-width
 * types are required, the corresponding integer constant macros are defined
 * here. This implementation also defines minimum-width types for every other
 * integer width that the target implements, so corresponding macros are
 * defined below, too.
 *
 * These macros are defined using the same successive-shrinking approach as
 * the type definitions above. It is likewise important that macros are defined
 * in order of decending width.
 *
 * Note that C++ should not check __STDC_CONSTANT_MACROS here, contrary to the
 * claims of the C standard (see C++ 18.3.1p2, [cstdint.syn]).
 */

#define __int_c_join(a, b) a##b
#define __int_c(v, suffix) __int_c_join(v, suffix)
#define __uint_c(v, suffix) __int_c_join(v##U, suffix)

#ifdef __INT64_TYPE__
#ifdef __INT64_C_SUFFIX__
#define __int64_c_suffix __INT64_C_SUFFIX__
#else
#undef __int64_c_suffix
#endif /* __INT64_C_SUFFIX__ */
#endif /* __INT64_TYPE__ */

#ifdef __int_least64_t
#ifdef __int64_c_suffix
#define INT64_C(v) __int_c(v, __int64_c_suffix)
#define UINT64_C(v) __uint_c(v, __int64_c_suffix)
#else
#define INT64_C(v) v
#define UINT64_C(v) v##U
#endif /* __int64_c_suffix */
#endif /* __int_least64_t */


#ifdef __INT32_TYPE__
#ifdef __INT32_C_SUFFIX__
#define __int32_c_suffix __INT32_C_SUFFIX__
#else
#undef __int32_c_suffix
#endif /* __INT32_C_SUFFIX__ */
#endif /* __INT32_TYPE__ */

#ifdef __int_least32_t
#ifdef __int32_c_suffix
#define INT32_C(v) __int_c(v, __int32_c_suffix)
#define UINT32_C(v) __uint_c(v, __int32_c_suffix)
#else
#define INT32_C(v) v
#define UINT32_C(v) v##U
#endif /* __int32_c_suffix */
#endif /* __int_least32_t */

#ifdef __INT16_TYPE__
#ifdef __INT16_C_SUFFIX__
#define __int16_c_suffix __INT16_C_SUFFIX__
#else
#undef __int16_c_suffix
#endif /* __INT16_C_SUFFIX__ */
#endif /* __INT16_TYPE__ */

#ifdef __int_least16_t
#ifdef __int16_c_suffix
#define INT16_C(v) __int_c(v, __int16_c_suffix)
#define UINT16_C(v) __uint_c(v, __int16_c_suffix)
#else
#define INT16_C(v) v
#define UINT16_C(v) v##U
#endif /* __int16_c_suffix */
#endif /* __int_least16_t */


#ifdef __INT8_TYPE__
#ifdef __INT8_C_SUFFIX__
#define __int8_c_suffix __INT8_C_SUFFIX__
#else
#undef __int8_c_suffix
#endif /* __INT8_C_SUFFIX__ */
#endif /* __INT8_TYPE__ */

#ifdef __int_least8_t
#ifdef __int8_c_suffix
#define INT8_C(v) __int_c(v, __int8_c_suffix)
#define UINT8_C(v) __uint_c(v, __int8_c_suffix)
#else
#define INT8_C(v) v
#define UINT8_C(v) v##U
#endif /* __int8_c_suffix */
#endif /* __int_least8_t */

/* C99 7.18.2.1 Limits of exact-width integer types.
 * C99 7.18.2.2 Limits of minimum-width integer types.
 * C99 7.18.2.3 Limits of fastest minimum-width integer types.
 *
 * The presence of limit macros are completely optional in C99.  This
 * implementation defines limits for all of the types (exact- and
 * minimum-width) that it defines above, using the limits of the minimum-width
 * type for any types that do not have exact-width representations.
 *
 * As in the type definitions, this section takes an approach of
 * successive-shrinking to determine which limits to use for the standard (8,
 * 16, 32, 64) bit widths when they don't have exact representations. It is
 * therefore important that the definitions be kept in order of decending
 * widths.
 *
 * Note that C++ should not check __STDC_LIMIT_MACROS here, contrary to the
 * claims of the C standard (see C++ 18.3.1p2, [cstdint.syn]).
 */

#ifdef __INT64_TYPE__
#define INT64_MAX INT64_C(9223372036854775807)
#define INT64_MIN (-INT64_C(9223372036854775807) - 1)
#define UINT64_MAX UINT64_C(18446744073709551615)
#define __INT_LEAST64_MIN INT64_MIN
#define __INT_LEAST64_MAX INT64_MAX
#define __UINT_LEAST64_MAX UINT64_MAX
#endif /* __INT64_TYPE__ */

#ifdef __INT_LEAST64_MIN
#define INT_LEAST64_MIN __INT_LEAST64_MIN
#define INT_LEAST64_MAX __INT_LEAST64_MAX
#define UINT_LEAST64_MAX __UINT_LEAST64_MAX
#define INT_FAST64_MIN __INT_LEAST64_MIN
#define INT_FAST64_MAX __INT_LEAST64_MAX
#define UINT_FAST64_MAX __UINT_LEAST64_MAX
#endif /* __INT_LEAST64_MIN */

#ifdef __INT32_TYPE__
#define INT32_MAX INT32_C(2147483647)
#define INT32_MIN (-INT32_C(2147483647) - 1)
#define UINT32_MAX UINT32_C(4294967295)
#define __INT_LEAST32_MIN INT32_MIN
#define __INT_LEAST32_MAX INT32_MAX
#define __UINT_LEAST32_MAX UINT32_MAX
#endif /* __INT32_TYPE__ */

#ifdef __INT_LEAST32_MIN
#define INT_LEAST32_MIN __INT_LEAST32_MIN
#define INT_LEAST32_MAX __INT_LEAST32_MAX
#define UINT_LEAST32_MAX __UINT_LEAST32_MAX
#define INT_FAST32_MIN __INT_LEAST32_MIN
#define INT_FAST32_MAX __INT_LEAST32_MAX
#define UINT_FAST32_MAX __UINT_LEAST32_MAX
#endif /* __INT_LEAST32_MIN */

#ifdef __INT16_TYPE__
#define INT16_MAX INT16_C(32767)
#define INT16_MIN (-INT16_C(32767) - 1)
#define UINT16_MAX UINT16_C(65535)
#define __INT_LEAST16_MIN INT16_MIN
#define __INT_LEAST16_MAX INT16_MAX
#define __UINT_LEAST16_MAX UINT16_MAX
#endif /* __INT16_TYPE__ */

#ifdef __INT_LEAST16_MIN
#define INT_LEAST16_MIN __INT_LEAST16_MIN
#define INT_LEAST16_MAX __INT_LEAST16_MAX
#define UINT_LEAST16_MAX __UINT_LEAST16_MAX
#define INT_FAST16_MIN __INT_LEAST16_MIN
#define INT_FAST16_MAX __INT_LEAST16_MAX
#define UINT_FAST16_MAX __UINT_LEAST16_MAX
#endif /* __INT_LEAST16_MIN */


#ifdef __INT8_TYPE__
#define INT8_MAX INT8_C(127)
#define INT8_MIN (-INT8_C(127) - 1)
#define UINT8_MAX UINT8_C(255)
#define __INT_LEAST8_MIN INT8_MIN
#define __INT_LEAST8_MAX INT8_MAX
#define __UINT_LEAST8_MAX UINT8_MAX
#endif /* __INT8_TYPE__ */

#ifdef __INT_LEAST8_MIN
#define INT_LEAST8_MIN __INT_LEAST8_MIN
#define INT_LEAST8_MAX __INT_LEAST8_MAX
#define UINT_LEAST8_MAX __UINT_LEAST8_MAX
#define INT_FAST8_MIN __INT_LEAST8_MIN
#define INT_FAST8_MAX __INT_LEAST8_MAX
#define UINT_FAST8_MAX __UINT_LEAST8_MAX
#endif /* __INT_LEAST8_MIN */

/* Some utility macros */
#define __INTN_MIN(n) __stdint_join3(INT, n, _MIN)
#define __INTN_MAX(n) __stdint_join3(INT, n, _MAX)
#define __UINTN_MAX(n) __stdint_join3(UINT, n, _MAX)
#define __INTN_C(n, v) __stdint_join3(INT, n, _C(v))
#define __UINTN_C(n, v) __stdint_join3(UINT, n, _C(v))

/* C99 7.18.2.4 Limits of integer types capable of holding object pointers. */
/* C99 7.18.3 Limits of other integer types. */

#define INTPTR_MIN (-__INTPTR_MAX__ - 1)
#define INTPTR_MAX __INTPTR_MAX__
#define UINTPTR_MAX __UINTPTR_MAX__
#define PTRDIFF_MIN (-__PTRDIFF_MAX__ - 1)
#define PTRDIFF_MAX __PTRDIFF_MAX__
#define SIZE_MAX __SIZE_MAX__

/* ISO9899:2011 7.20 (C11 Annex K): Define RSIZE_MAX if __STDC_WANT_LIB_EXT1__
 * is enabled. */
#if defined(__STDC_WANT_LIB_EXT1__) && __STDC_WANT_LIB_EXT1__ >= 1
#define RSIZE_MAX (SIZE_MAX >> 1)
#endif

/* C99 7.18.2.5 Limits of greatest-width integer types. */
#define INTMAX_MIN (-__INTMAX_MAX__ - 1)
#define INTMAX_MAX __INTMAX_MAX__
#define UINTMAX_MAX __UINTMAX_MAX__

/* C99 7.18.3 Limits of other integer types. */
#define SIG_ATOMIC_MIN __INTN_MIN(__SIG_ATOMIC_WIDTH__)
#define SIG_ATOMIC_MAX __INTN_MAX(__SIG_ATOMIC_WIDTH__)
#ifdef __WINT_UNSIGNED__
#define WINT_MIN __UINTN_C(__WINT_WIDTH__, 0)
#define WINT_MAX __UINTN_MAX(__WINT_WIDTH__)
#else
#define WINT_MIN __INTN_MIN(__WINT_WIDTH__)
#define WINT_MAX __INTN_MAX(__WINT_WIDTH__)
#endif

#ifndef WCHAR_MAX
#define WCHAR_MAX __WCHAR_MAX__
#endif
#ifndef WCHAR_MIN
#if __WCHAR_MAX__ == __INTN_MAX(__WCHAR_WIDTH__)
#define WCHAR_MIN __INTN_MIN(__WCHAR_WIDTH__)
#else
#define WCHAR_MIN __UINTN_C(__WCHAR_WIDTH__, 0)
#endif
#endif

/* 7.18.4.2 Macros for greatest-width integer constants. */
#define INTMAX_C(v) __int_c(v, __INTMAX_C_SUFFIX__)
#define UINTMAX_C(v) __int_c(v, __UINTMAX_C_SUFFIX__)


#endif // SLANG_LLVM_H

#else // SLANG_LLVM
#if SLANG_GCC_FAMILY && __GNUC__ < 6
#include <cmath>
#define SLANG_PRELUDE_STD std::
#else
#include <math.h>
#define SLANG_PRELUDE_STD
#endif

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#endif // SLANG_LLVM

// Is intptr_t not equal to equal-width sized integer type?
#if defined(__APPLE__)
#define SLANG_INTPTR_TYPE_IS_DISTINCT 1
#else
#define SLANG_INTPTR_TYPE_IS_DISTINCT 0
#endif

#if defined(_MSC_VER)
#define SLANG_PRELUDE_SHARED_LIB_EXPORT __declspec(dllexport)
#else
#define SLANG_PRELUDE_SHARED_LIB_EXPORT __attribute__((__visibility__("default")))
// #   define SLANG_PRELUDE_SHARED_LIB_EXPORT __attribute__ ((dllexport))
// __attribute__((__visibility__("default")))
#endif

#ifdef __cplusplus
#define SLANG_PRELUDE_EXTERN_C extern "C"
#define SLANG_PRELUDE_EXTERN_C_START \
    extern "C"                       \
    {
#define SLANG_PRELUDE_EXTERN_C_END }
#else
#define SLANG_PRELUDE_EXTERN_C
#define SLANG_PRELUDE_EXTERN_C_START
#define SLANG_PRELUDE_EXTERN_C_END
#endif

#define SLANG_PRELUDE_EXPORT SLANG_PRELUDE_EXTERN_C SLANG_PRELUDE_SHARED_LIB_EXPORT
#define SLANG_PRELUDE_EXPORT_START SLANG_PRELUDE_EXTERN_C_START SLANG_PRELUDE_SHARED_LIB_EXPORT
#define SLANG_PRELUDE_EXPORT_END SLANG_PRELUDE_EXTERN_C_END

#ifndef INFINITY
// Must overflow for double
#define INFINITY float(1e+300 * 1e+300)
#endif

#ifndef SLANG_INFINITY
#define SLANG_INFINITY INFINITY
#endif

// Detect the compiler type

#ifndef SLANG_COMPILER
#define SLANG_COMPILER

/*
Compiler defines, see http://sourceforge.net/p/predef/wiki/Compilers/
NOTE that SLANG_VC holds the compiler version - not just 1 or 0
*/
#if defined(_MSC_VER)
#if _MSC_VER >= 1900
#define SLANG_VC 14
#elif _MSC_VER >= 1800
#define SLANG_VC 12
#elif _MSC_VER >= 1700
#define SLANG_VC 11
#elif _MSC_VER >= 1600
#define SLANG_VC 10
#elif _MSC_VER >= 1500
#define SLANG_VC 9
#else
#error "unknown version of Visual C++ compiler"
#endif
#elif defined(__clang__)
#define SLANG_CLANG 1
#elif defined(__SNC__)
#define SLANG_SNC 1
#elif defined(__ghs__)
#define SLANG_GHS 1
#elif defined(__GNUC__) /* note: __clang__, __SNC__, or __ghs__ imply __GNUC__ */
#define SLANG_GCC 1
#else
#error "unknown compiler"
#endif
/*
Any compilers not detected by the above logic are now now explicitly zeroed out.
*/
#ifndef SLANG_VC
#define SLANG_VC 0
#endif
#ifndef SLANG_CLANG
#define SLANG_CLANG 0
#endif
#ifndef SLANG_SNC
#define SLANG_SNC 0
#endif
#ifndef SLANG_GHS
#define SLANG_GHS 0
#endif
#ifndef SLANG_GCC
#define SLANG_GCC 0
#endif
#endif /* SLANG_COMPILER */

/*
The following section attempts to detect the target platform being compiled for.

If an application defines `SLANG_PLATFORM` before including this header,
they take responsibility for setting any compiler-dependent macros
used later in the file.

Most applications should not need to touch this section.
*/
#ifndef SLANG_PLATFORM
#define SLANG_PLATFORM
/**
Operating system defines, see http://sourceforge.net/p/predef/wiki/OperatingSystems/
*/
#if defined(WINAPI_FAMILY) && WINAPI_FAMILY == WINAPI_PARTITION_APP
#define SLANG_WINRT 1 /* Windows Runtime, either on Windows RT or Windows 8 */
#elif defined(XBOXONE)
#define SLANG_XBOXONE 1
#elif defined(_WIN64) /* note: XBOXONE implies _WIN64 */
#define SLANG_WIN64 1
#elif defined(_M_PPC)
#define SLANG_X360 1
#elif defined(_WIN32) /* note: _M_PPC implies _WIN32 */
#define SLANG_WIN32 1
#elif defined(__ANDROID__)
#define SLANG_ANDROID 1
#elif defined(__linux__) || defined(__CYGWIN__) /* note: __ANDROID__ implies __linux__ */
#define SLANG_LINUX 1
#elif defined(__APPLE__) && !defined(SLANG_LLVM)
#include "TargetConditionals.h"
#if TARGET_OS_MAC
#define SLANG_OSX 1
#else
#define SLANG_IOS 1
#endif
#elif defined(__APPLE__)
// On `slang-llvm` we can't inclue "TargetConditionals.h" in general, so for now assume its
// OSX.
#define SLANG_OSX 1
#elif defined(__CELLOS_LV2__)
#define SLANG_PS3 1
#elif defined(__ORBIS__)
#define SLANG_PS4 1
#elif defined(__SNC__) && defined(__arm__)
#define SLANG_PSP2 1
#elif defined(__ghs__)
#define SLANG_WIIU 1
#else
#error "unknown target platform"
#endif


/*
Any platforms not detected by the above logic are now now explicitly zeroed out.
*/
#ifndef SLANG_WINRT
#define SLANG_WINRT 0
#endif
#ifndef SLANG_XBOXONE
#define SLANG_XBOXONE 0
#endif
#ifndef SLANG_WIN64
#define SLANG_WIN64 0
#endif
#ifndef SLANG_X360
#define SLANG_X360 0
#endif
#ifndef SLANG_WIN32
#define SLANG_WIN32 0
#endif
#ifndef SLANG_ANDROID
#define SLANG_ANDROID 0
#endif
#ifndef SLANG_LINUX
#define SLANG_LINUX 0
#endif
#ifndef SLANG_IOS
#define SLANG_IOS 0
#endif
#ifndef SLANG_OSX
#define SLANG_OSX 0
#endif
#ifndef SLANG_PS3
#define SLANG_PS3 0
#endif
#ifndef SLANG_PS4
#define SLANG_PS4 0
#endif
#ifndef SLANG_PSP2
#define SLANG_PSP2 0
#endif
#ifndef SLANG_WIIU
#define SLANG_WIIU 0
#endif
#endif /* SLANG_PLATFORM */

/* Shorthands for "families" of compilers/platforms */
#define SLANG_GCC_FAMILY (SLANG_CLANG || SLANG_SNC || SLANG_GHS || SLANG_GCC)
#define SLANG_WINDOWS_FAMILY (SLANG_WINRT || SLANG_WIN32 || SLANG_WIN64)
#define SLANG_MICROSOFT_FAMILY (SLANG_XBOXONE || SLANG_X360 || SLANG_WINDOWS_FAMILY)
#define SLANG_LINUX_FAMILY (SLANG_LINUX || SLANG_ANDROID)
#define SLANG_APPLE_FAMILY (SLANG_IOS || SLANG_OSX) /* equivalent to #if __APPLE__ */
#define SLANG_UNIX_FAMILY \
    (SLANG_LINUX_FAMILY || SLANG_APPLE_FAMILY) /* shortcut for unix/posix platforms */

// GCC Specific
#if SLANG_GCC_FAMILY

#if INTPTR_MAX == INT64_MAX
#define SLANG_64BIT 1
#else
#define SLANG_64BIT 0
#endif

#define SLANG_BREAKPOINT(id) __builtin_trap()

// Use this macro instead of offsetof, because gcc produces warning if offsetof is used on a
// non POD type, even though it produces the correct result
#define SLANG_OFFSET_OF(T, ELEMENT) (size_t(&((T*)1)->ELEMENT) - 1)
#endif // SLANG_GCC_FAMILY

// Microsoft VC specific
#if SLANG_VC

#define SLANG_BREAKPOINT(id) __debugbreak();

#endif // SLANG_VC

// Default impls

#ifndef SLANG_OFFSET_OF
#define SLANG_OFFSET_OF(X, Y) offsetof(X, Y)
#endif

#ifndef SLANG_BREAKPOINT
// Make it crash with a write to 0!
#define SLANG_BREAKPOINT(id) (*((int*)0) = int(id));
#endif

// If slang.h has been included we don't need any of these definitions
#ifndef SLANG_H

/* Macro for declaring if a method is no throw. Should be set before the return parameter. */
#ifndef SLANG_NO_THROW
#if SLANG_WINDOWS_FAMILY && !defined(SLANG_DISABLE_EXCEPTIONS)
#define SLANG_NO_THROW __declspec(nothrow)
#endif
#endif
#ifndef SLANG_NO_THROW
#define SLANG_NO_THROW
#endif

/* The `SLANG_STDCALL` and `SLANG_MCALL` defines are used to set the calling
convention for interface methods.
*/
#ifndef SLANG_STDCALL
#if SLANG_MICROSOFT_FAMILY
#define SLANG_STDCALL __stdcall
#else
#define SLANG_STDCALL
#endif
#endif
#ifndef SLANG_MCALL
#define SLANG_MCALL SLANG_STDCALL
#endif

#ifndef SLANG_FORCE_INLINE
#define SLANG_FORCE_INLINE inline
#endif

// TODO(JS): Should these be in slang-cpp-types.h?
// They are more likely to clash with slang.h

struct SlangUUID
{
    uint32_t data1;
    uint16_t data2;
    uint16_t data3;
    uint8_t data4[8];
};

typedef int32_t SlangResult;

struct ISlangUnknown
{
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    queryInterface(SlangUUID const& uuid, void** outObject) = 0;
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL addRef() = 0;
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL release() = 0;
};

#define SLANG_COM_INTERFACE(a, b, c, d0, d1, d2, d3, d4, d5, d6, d7)             \
public:                                                                          \
    SLANG_FORCE_INLINE static const SlangUUID& getTypeGuid()                     \
    {                                                                            \
        static const SlangUUID guid = {a, b, c, d0, d1, d2, d3, d4, d5, d6, d7}; \
        return guid;                                                             \
    }
#endif // SLANG_H

// Includes

#ifndef SLANG_PRELUDE_SCALAR_INTRINSICS_H
#define SLANG_PRELUDE_SCALAR_INTRINSICS_H

#if !defined(SLANG_LLVM) && SLANG_PROCESSOR_X86_64 && SLANG_VC
//  If we have visual studio and 64 bit processor, we can assume we have popcnt, and can include
//  x86 intrinsics
#include <intrin.h>
#endif

#ifndef SLANG_FORCE_INLINE
#define SLANG_FORCE_INLINE inline
#endif

#ifdef SLANG_PRELUDE_NAMESPACE
namespace SLANG_PRELUDE_NAMESPACE
{
#endif

#ifndef SLANG_PRELUDE_PI
#define SLANG_PRELUDE_PI 3.14159265358979323846
#endif

union Union32
{
    uint32_t u;
    int32_t i;
    float f;
};

union Union64
{
    uint64_t u;
    int64_t i;
    double d;
};

// 32 bit cast conversions
SLANG_FORCE_INLINE int32_t _bitCastFloatToInt(float f)
{
    Union32 u;
    u.f = f;
    return u.i;
}
SLANG_FORCE_INLINE float _bitCastIntToFloat(int32_t i)
{
    Union32 u;
    u.i = i;
    return u.f;
}
SLANG_FORCE_INLINE uint32_t _bitCastFloatToUInt(float f)
{
    Union32 u;
    u.f = f;
    return u.u;
}
SLANG_FORCE_INLINE float _bitCastUIntToFloat(uint32_t ui)
{
    Union32 u;
    u.u = ui;
    return u.f;
}

// ----------------------------- F32 -----------------------------------------

// Helpers
SLANG_FORCE_INLINE float F32_calcSafeRadians(float radians);

#ifdef SLANG_LLVM

SLANG_PRELUDE_EXTERN_C_START

// Unary
float F32_ceil(float f);
float F32_floor(float f);
float F32_round(float f);
float F32_sin(float f);
float F32_cos(float f);
float F32_tan(float f);
float F32_asin(float f);
float F32_acos(float f);
float F32_atan(float f);
float F32_sinh(float f);
float F32_cosh(float f);
float F32_tanh(float f);
float F32_asinh(float f);
float F32_acosh(float f);
float F32_atanh(float f);
float F32_log2(float f);
float F32_log(float f);
float F32_log10(float f);
float F32_exp2(float f);
float F32_exp(float f);
float F32_abs(float f);
float F32_trunc(float f);
float F32_sqrt(float f);

bool F32_isnan(float f);
bool F32_isfinite(float f);
bool F32_isinf(float f);

// Binary
SLANG_FORCE_INLINE float F32_min(float a, float b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE float F32_max(float a, float b)
{
    return a > b ? a : b;
}
float F32_pow(float a, float b);
float F32_fmod(float a, float b);
float F32_remainder(float a, float b);
float F32_atan2(float a, float b);

float F32_frexp(float x, int* e);

float F32_modf(float x, float* ip);

// Ternary
SLANG_FORCE_INLINE float F32_fma(float a, float b, float c)
{
    return a * b + c;
}

SLANG_PRELUDE_EXTERN_C_END

#else

// Unary
SLANG_FORCE_INLINE float F32_ceil(float f)
{
    return ::ceilf(f);
}
SLANG_FORCE_INLINE float F32_floor(float f)
{
    return ::floorf(f);
}
SLANG_FORCE_INLINE float F32_round(float f)
{
    return ::roundf(f);
}
SLANG_FORCE_INLINE float F32_sin(float f)
{
    return ::sinf(f);
}
SLANG_FORCE_INLINE float F32_cos(float f)
{
    return ::cosf(f);
}
SLANG_FORCE_INLINE float F32_tan(float f)
{
    return ::tanf(f);
}
SLANG_FORCE_INLINE float F32_asin(float f)
{
    return ::asinf(f);
}
SLANG_FORCE_INLINE float F32_acos(float f)
{
    return ::acosf(f);
}
SLANG_FORCE_INLINE float F32_atan(float f)
{
    return ::atanf(f);
}
SLANG_FORCE_INLINE float F32_sinh(float f)
{
    return ::sinhf(f);
}
SLANG_FORCE_INLINE float F32_cosh(float f)
{
    return ::coshf(f);
}
SLANG_FORCE_INLINE float F32_tanh(float f)
{
    return ::tanhf(f);
}
SLANG_FORCE_INLINE float F32_asinh(float f)
{
    return ::asinhf(f);
}
SLANG_FORCE_INLINE float F32_acosh(float f)
{
    return ::acoshf(f);
}
SLANG_FORCE_INLINE float F32_atanh(float f)
{
    return ::atanhf(f);
}
SLANG_FORCE_INLINE float F32_log2(float f)
{
    return ::log2f(f);
}
SLANG_FORCE_INLINE float F32_log(float f)
{
    return ::logf(f);
}
SLANG_FORCE_INLINE float F32_log10(float f)
{
    return ::log10f(f);
}
SLANG_FORCE_INLINE float F32_exp2(float f)
{
    return ::exp2f(f);
}
SLANG_FORCE_INLINE float F32_exp(float f)
{
    return ::expf(f);
}
SLANG_FORCE_INLINE float F32_abs(float f)
{
    return ::fabsf(f);
}
SLANG_FORCE_INLINE float F32_trunc(float f)
{
    return ::truncf(f);
}
SLANG_FORCE_INLINE float F32_sqrt(float f)
{
    return ::sqrtf(f);
}

SLANG_FORCE_INLINE bool F32_isnan(float f)
{
    return SLANG_PRELUDE_STD isnan(f);
}
SLANG_FORCE_INLINE bool F32_isfinite(float f)
{
    return SLANG_PRELUDE_STD isfinite(f);
}
SLANG_FORCE_INLINE bool F32_isinf(float f)
{
    return SLANG_PRELUDE_STD isinf(f);
}

// Binary
SLANG_FORCE_INLINE float F32_min(float a, float b)
{
    return ::fminf(a, b);
}
SLANG_FORCE_INLINE float F32_max(float a, float b)
{
    return ::fmaxf(a, b);
}
SLANG_FORCE_INLINE float F32_pow(float a, float b)
{
    return ::powf(a, b);
}
SLANG_FORCE_INLINE float F32_fmod(float a, float b)
{
    return ::fmodf(a, b);
}
SLANG_FORCE_INLINE float F32_remainder(float a, float b)
{
    return ::remainderf(a, b);
}
SLANG_FORCE_INLINE float F32_atan2(float a, float b)
{
    return float(::atan2(a, b));
}

SLANG_FORCE_INLINE float F32_frexp(float x, int* e)
{
    return ::frexpf(x, e);
}

SLANG_FORCE_INLINE float F32_modf(float x, float* ip)
{
    return ::modff(x, ip);
}

// Ternary
SLANG_FORCE_INLINE float F32_fma(float a, float b, float c)
{
    return ::fmaf(a, b, c);
}

#endif

SLANG_FORCE_INLINE float F32_calcSafeRadians(float radians)
{
    // Put 0 to 2pi cycles to cycle around 0 to 1
    float a = radians * (1.0f / float(SLANG_PRELUDE_PI * 2));
    // Get truncated fraction, as value in  0 - 1 range
    a = a - F32_floor(a);
    // Convert back to 0 - 2pi range
    return (a * float(SLANG_PRELUDE_PI * 2));
}

SLANG_FORCE_INLINE float F32_rsqrt(float f)
{
    return 1.0f / F32_sqrt(f);
}
SLANG_FORCE_INLINE int F32_sign(float f)
{
    return (f == 0.0f) ? 0 : ((f < 0.0f) ? -1 : 1);
}
SLANG_FORCE_INLINE float F32_frac(float f)
{
    return f - F32_floor(f);
}

SLANG_FORCE_INLINE uint32_t F32_asuint(float f)
{
    Union32 u;
    u.f = f;
    return u.u;
}
SLANG_FORCE_INLINE int32_t F32_asint(float f)
{
    Union32 u;
    u.f = f;
    return u.i;
}

// ----------------------------- F64 -----------------------------------------

SLANG_FORCE_INLINE double F64_calcSafeRadians(double radians);

#ifdef SLANG_LLVM

SLANG_PRELUDE_EXTERN_C_START

// Unary
double F64_ceil(double f);
double F64_floor(double f);
double F64_round(double f);
double F64_sin(double f);
double F64_cos(double f);
double F64_tan(double f);
double F64_asin(double f);
double F64_acos(double f);
double F64_atan(double f);
double F64_sinh(double f);
double F64_cosh(double f);
double F64_tanh(double f);
double F64_asinh(double f);
double F64_acosh(double f);
double F64_atanh(double f);
double F64_log2(double f);
double F64_log(double f);
double F64_log10(double f);
double F64_exp2(double f);
double F64_exp(double f);
double F64_abs(double f);
double F64_trunc(double f);
double F64_sqrt(double f);

bool F64_isnan(double f);
bool F64_isfinite(double f);
bool F64_isinf(double f);

// Binary
SLANG_FORCE_INLINE double F64_min(double a, double b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE double F64_max(double a, double b)
{
    return a > b ? a : b;
}
double F64_pow(double a, double b);
double F64_fmod(double a, double b);
double F64_remainder(double a, double b);
double F64_atan2(double a, double b);

double F64_frexp(double x, int* e);

double F64_modf(double x, double* ip);

// Ternary
SLANG_FORCE_INLINE double F64_fma(double a, double b, double c)
{
    return a * b + c;
}

SLANG_PRELUDE_EXTERN_C_END

#else // SLANG_LLVM

// Unary
SLANG_FORCE_INLINE double F64_ceil(double f)
{
    return ::ceil(f);
}
SLANG_FORCE_INLINE double F64_floor(double f)
{
    return ::floor(f);
}
SLANG_FORCE_INLINE double F64_round(double f)
{
    return ::round(f);
}
SLANG_FORCE_INLINE double F64_sin(double f)
{
    return ::sin(f);
}
SLANG_FORCE_INLINE double F64_cos(double f)
{
    return ::cos(f);
}
SLANG_FORCE_INLINE double F64_tan(double f)
{
    return ::tan(f);
}
SLANG_FORCE_INLINE double F64_asin(double f)
{
    return ::asin(f);
}
SLANG_FORCE_INLINE double F64_acos(double f)
{
    return ::acos(f);
}
SLANG_FORCE_INLINE double F64_atan(double f)
{
    return ::atan(f);
}
SLANG_FORCE_INLINE double F64_sinh(double f)
{
    return ::sinh(f);
}
SLANG_FORCE_INLINE double F64_cosh(double f)
{
    return ::cosh(f);
}
SLANG_FORCE_INLINE double F64_tanh(double f)
{
    return ::tanh(f);
}
SLANG_FORCE_INLINE double F64_log2(double f)
{
    return ::log2(f);
}
SLANG_FORCE_INLINE double F64_log(double f)
{
    return ::log(f);
}
SLANG_FORCE_INLINE double F64_log10(float f)
{
    return ::log10(f);
}
SLANG_FORCE_INLINE double F64_exp2(double f)
{
    return ::exp2(f);
}
SLANG_FORCE_INLINE double F64_exp(double f)
{
    return ::exp(f);
}
SLANG_FORCE_INLINE double F64_abs(double f)
{
    return ::fabs(f);
}
SLANG_FORCE_INLINE double F64_trunc(double f)
{
    return ::trunc(f);
}
SLANG_FORCE_INLINE double F64_sqrt(double f)
{
    return ::sqrt(f);
}


SLANG_FORCE_INLINE bool F64_isnan(double f)
{
    return SLANG_PRELUDE_STD isnan(f);
}
SLANG_FORCE_INLINE bool F64_isfinite(double f)
{
    return SLANG_PRELUDE_STD isfinite(f);
}
SLANG_FORCE_INLINE bool F64_isinf(double f)
{
    return SLANG_PRELUDE_STD isinf(f);
}

// Binary
SLANG_FORCE_INLINE double F64_min(double a, double b)
{
    return ::fmin(a, b);
}
SLANG_FORCE_INLINE double F64_max(double a, double b)
{
    return ::fmax(a, b);
}
SLANG_FORCE_INLINE double F64_pow(double a, double b)
{
    return ::pow(a, b);
}
SLANG_FORCE_INLINE double F64_fmod(double a, double b)
{
    return ::fmod(a, b);
}
SLANG_FORCE_INLINE double F64_remainder(double a, double b)
{
    return ::remainder(a, b);
}
SLANG_FORCE_INLINE double F64_atan2(double a, double b)
{
    return ::atan2(a, b);
}

SLANG_FORCE_INLINE double F64_frexp(double x, int* e)
{
    return ::frexp(x, e);
}

SLANG_FORCE_INLINE double F64_modf(double x, double* ip)
{
    return ::modf(x, ip);
}

// Ternary
SLANG_FORCE_INLINE double F64_fma(double a, double b, double c)
{
    return ::fma(a, b, c);
}

#endif // SLANG_LLVM

SLANG_FORCE_INLINE double F64_rsqrt(double f)
{
    return 1.0 / F64_sqrt(f);
}
SLANG_FORCE_INLINE int F64_sign(double f)
{
    return (f == 0.0) ? 0 : ((f < 0.0) ? -1 : 1);
}
SLANG_FORCE_INLINE double F64_frac(double f)
{
    return f - F64_floor(f);
}

SLANG_FORCE_INLINE void F64_asuint(double d, uint32_t* low, uint32_t* hi)
{
    Union64 u;
    u.d = d;
    *low = uint32_t(u.u);
    *hi = uint32_t(u.u >> 32);
}

SLANG_FORCE_INLINE void F64_asint(double d, int32_t* low, int32_t* hi)
{
    Union64 u;
    u.d = d;
    *low = int32_t(u.u);
    *hi = int32_t(u.u >> 32);
}

SLANG_FORCE_INLINE double F64_calcSafeRadians(double radians)
{
    // Put 0 to 2pi cycles to cycle around 0 to 1
    double a = radians * (1.0f / (SLANG_PRELUDE_PI * 2));
    // Get truncated fraction, as value in  0 - 1 range
    a = a - F64_floor(a);
    // Convert back to 0 - 2pi range
    return (a * (SLANG_PRELUDE_PI * 2));
}

// ----------------------------- F16 -----------------------------------------

// This impl is based on FloatToHalf that is in Slang codebase
SLANG_FORCE_INLINE uint32_t f32tof16(const float value)
{
    const uint32_t inBits = _bitCastFloatToUInt(value);

    // bits initially set to just the sign bit
    uint32_t bits = (inBits >> 16) & 0x8000;
    // Mantissa can't be used as is, as it holds last bit, for rounding.
    uint32_t m = (inBits >> 12) & 0x07ff;
    uint32_t e = (inBits >> 23) & 0xff;

    if (e < 103)
    {
        // It's zero
        return bits;
    }
    if (e == 0xff)
    {
        // Could be a NAN or INF. Is INF if *input* mantissa is 0.

        // Remove last bit for rounding to make output mantissa.
        m >>= 1;

        // We *assume* float16/float32 signaling bit and remaining bits
        // semantics are the same. (The signalling bit convention is target specific!).
        // Non signal bit's usage within mantissa for a NAN are also target specific.

        // If the m is 0, it could be because the result is INF, but it could also be because all
        // the bits that made NAN were dropped as we have less mantissa bits in f16.

        // To fix for this we make non zero if m is 0 and the input mantissa was not.
        // This will (typically) produce a signalling NAN.
        m += uint32_t(m == 0 && (inBits & 0x007fffffu));

        // Combine for output
        return (bits | 0x7c00u | m);
    }
    if (e > 142)
    {
        // INF.
        return bits | 0x7c00u;
    }
    if (e < 113)
    {
        m |= 0x0800u;
        bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1);
        return bits;
    }
    bits |= ((e - 112) << 10) | (m >> 1);
    bits += m & 1;
    return bits;
}

static const float g_f16tof32Magic = _bitCastIntToFloat((127 + (127 - 15)) << 23);

SLANG_FORCE_INLINE float f16tof32(const uint32_t value)
{
    const uint32_t sign = (value & 0x8000) << 16;
    uint32_t exponent = (value & 0x7c00) >> 10;
    uint32_t mantissa = (value & 0x03ff);

    if (exponent == 0)
    {
        // If mantissa is 0 we are done, as output is 0.
        // If it's not zero we must have a denormal.
        if (mantissa)
        {
            // We have a denormal so use the magic to do exponent adjust
            return _bitCastIntToFloat(sign | ((value & 0x7fff) << 13)) * g_f16tof32Magic;
        }
    }
    else
    {
        // If the exponent is NAN or INF exponent is 0x1f on input.
        // If that's the case, we just need to set the exponent to 0xff on output
        // and the mantissa can just stay the same. If its 0 it's INF, else it is NAN and we just
        // copy the bits
        //
        // Else we need to correct the exponent in the normalized case.
        exponent = (exponent == 0x1F) ? 0xff : (exponent + (-15 + 127));
    }

    return _bitCastUIntToFloat(sign | (exponent << 23) | (mantissa << 13));
}

#ifndef SLANG_LLVM
#if __cplusplus >= 202302L
#include <stdfloat> // C++23
#else
// Define __STDC_WANT_IEC_60559_TYPES_EXT__ for compilers with reliable _Float16 support:
// - Clang 15+
// - GCC 12+
#if (defined(__clang__) && __clang_major__ >= 15) || (defined(__GNUC__) && __GNUC__ >= 12)
#ifndef __STDC_WANT_IEC_60559_TYPES_EXT__
#define __STDC_WANT_IEC_60559_TYPES_EXT__
#endif
#include <float.h>
#endif // __STDC_WANT_IEC_60559_TYPES_EXT__
#endif // (defined(__clang__) && __clang_major__ >= 15) || (defined(__GNUC__) && __GNUC__ >= 12)
#endif // C++23

#ifdef FLT16_MIN
typedef _Float16 half;
#elif __STDCPP_FLOAT16_T__ == 1
typedef std::float16_t half;
#else
uint32_t f32tof16(const float value);
float f16tof32(const uint32_t value);
struct half
{
    uint16_t data;

    half() = default;
    explicit half(float f) { store(f); }

    SLANG_FORCE_INLINE void store(float f) { data = f32tof16(f); }
    SLANG_FORCE_INLINE float load() const { return f16tof32(data); }

    half operator+(half other) const { return half(load() + other.load()); }
    half operator-(half other) const { return half(load() - other.load()); }
    half operator*(half other) const { return half(load() * other.load()); }
    half operator/(half other) const { return half(load() / other.load()); }
    half& operator+=(half other)
    {
        store(load() + other.load());
        return *this;
    }
    half& operator-=(half other)
    {
        store(load() - other.load());
        return *this;
    }
    half& operator*=(half other)
    {
        store(load() * other.load());
        return *this;
    }
    half& operator/=(half other)
    {
        store(load() / other.load());
        return *this;
    }

    bool operator<(half other) const { return load() < other.load(); }
    bool operator>(half other) const { return load() > other.load(); }
    bool operator<=(half other) const { return load() <= other.load(); }
    bool operator>=(half other) const { return load() >= other.load(); }
    bool operator==(half other) const { return load() == other.load(); }
    bool operator!=(half other) const { return load() != other.load(); }

    explicit operator float() const { return load(); }
};
#endif

half U16_ashalf(uint16_t x);

union Union16
{
    uint16_t u;
    int16_t i;
    half h;
};

SLANG_FORCE_INLINE uint16_t F16_asuint(half h)
{
    Union16 u;
    u.h = h;
    return u.u;
}

SLANG_FORCE_INLINE int16_t F16_asint(half h)
{
    Union16 u;
    u.h = h;
    return u.i;
}

SLANG_FORCE_INLINE half F16_ceil(half f)
{
    return half(F32_ceil(float(f)));
}

SLANG_FORCE_INLINE half F16_floor(half f)
{
    return half(F32_floor(float(f)));
}

SLANG_FORCE_INLINE half F16_round(half f)
{
    return half(F32_round(float(f)));
}

SLANG_FORCE_INLINE half F16_sin(half f)
{
    return half(F32_sin(float(f)));
}

SLANG_FORCE_INLINE half F16_cos(half f)
{
    return half(F32_cos(float(f)));
}

SLANG_FORCE_INLINE half F16_tan(half f)
{
    return half(F32_tan(float(f)));
}

SLANG_FORCE_INLINE half F16_asin(half f)
{
    return half(F32_asin(float(f)));
}

SLANG_FORCE_INLINE half F16_acos(half f)
{
    return half(F32_acos(float(f)));
}

SLANG_FORCE_INLINE half F16_atan(half f)
{
    return half(F32_atan(float(f)));
}

SLANG_FORCE_INLINE half F16_sinh(half f)
{
    return half(F32_sinh(float(f)));
}

SLANG_FORCE_INLINE half F16_cosh(half f)
{
    return half(F32_cosh(float(f)));
}

SLANG_FORCE_INLINE half F16_tanh(half f)
{
    return half(F32_tanh(float(f)));
}

SLANG_FORCE_INLINE half F16_asinh(half f)
{
    return half(F32_asinh(float(f)));
}

SLANG_FORCE_INLINE half F16_acosh(half f)
{
    return half(F32_acosh(float(f)));
}

SLANG_FORCE_INLINE half F16_atanh(half f)
{
    return half(F32_atanh(float(f)));
}

SLANG_FORCE_INLINE half F16_log2(half f)
{
    return half(F32_log2(float(f)));
}

SLANG_FORCE_INLINE half F16_log(half f)
{
    return half(F32_log(float(f)));
}

SLANG_FORCE_INLINE half F16_log10(half f)
{
    return half(F32_log10(float(f)));
}

SLANG_FORCE_INLINE half F16_exp2(half f)
{
    return half(F32_exp2(float(f)));
}

SLANG_FORCE_INLINE half F16_exp(half f)
{
    return half(F32_exp(float(f)));
}

SLANG_FORCE_INLINE half F16_abs(half f)
{
    return U16_ashalf(F16_asuint(f) & 0x7FFF);
}

SLANG_FORCE_INLINE half F16_trunc(half f)
{
    return half(F32_trunc(float(f)));
}

SLANG_FORCE_INLINE half F16_sqrt(half f)
{
    return half(F32_sqrt(float(f)));
}

SLANG_FORCE_INLINE bool F16_isnan(half f)
{
    uint16_t u = F16_asuint(f);
    return (u & 0x7C00) == 0x7C00 && (u & 0x3FF) != 0;
}

SLANG_FORCE_INLINE bool F16_isfinite(half f)
{
    uint16_t u = F16_asuint(f);
    return (u & 0x7C00) != 0x7C00;
}

SLANG_FORCE_INLINE bool F16_isinf(half f)
{
    uint16_t u = F16_asuint(f);
    return (u & 0x7C00) == 0x7C00 && (u & 0x3FF) == 0;
}

SLANG_FORCE_INLINE half F16_min(half a, half b)
{
    if (F16_isnan(a))
        return b;
    if (F16_isnan(b))
        return a;
    return a < b ? a : b;
}

SLANG_FORCE_INLINE half F16_max(half a, half b)
{
    if (F16_isnan(a))
        return b;
    if (F16_isnan(b))
        return a;
    return a > b ? a : b;
}

SLANG_FORCE_INLINE half F16_pow(half a, half b)
{
    return half(F32_pow(float(a), float(b)));
}

SLANG_FORCE_INLINE half F16_fmod(half a, half b)
{
    return half(F32_fmod(float(a), float(b)));
}

SLANG_FORCE_INLINE half F16_remainder(half a, half b)
{
    return half(F32_remainder(float(a), float(b)));
}

SLANG_FORCE_INLINE half F16_atan2(half a, half b)
{
    return half(F32_atan2(float(a), float(b)));
}

SLANG_FORCE_INLINE half F16_frexp(half x, int* e)
{
    return half(F32_frexp(float(x), e));
}

SLANG_FORCE_INLINE half F16_modf(half x, half* ip)
{
    float ipf;
    float res = F32_modf(float(x), &ipf);
    *ip = half(ipf);
    return half(res);
}

SLANG_FORCE_INLINE half F16_fma(half a, half b, half c)
{
    return half(F32_fma(float(a), float(b), float(c)));
}

SLANG_FORCE_INLINE half F16_calcSafeRadians(half radians)
{
    // Put 0 to 2pi cycles to cycle around 0 to 1
    float a = float(radians) * (1.0f / float(SLANG_PRELUDE_PI * 2));
    // Get truncated fraction, as value in  0 - 1 range
    a = a - F32_floor(a);
    // Convert back to 0 - 2pi range
    return half(a * float(SLANG_PRELUDE_PI * 2));
}

SLANG_FORCE_INLINE half F16_rsqrt(half f)
{
    return half(1.0f / F32_sqrt(float(f)));
}

SLANG_FORCE_INLINE int F16_sign(half f)
{
    uint16_t u = F16_asuint(f);
    if ((u & 0x7FFF) == 0)
        return 0;
    return (u & 0x8000) != 0 ? -1 : 1;
}

SLANG_FORCE_INLINE half F16_frac(half h)
{
    float f = float(h);
    return half(f - F32_floor(f));
}

// ----------------------------- U16 -----------------------------------------
SLANG_FORCE_INLINE uint32_t U16_countbits(uint16_t v)
{
#if SLANG_GCC_FAMILY && !defined(SLANG_LLVM)
    return __builtin_popcount(uint32_t(v));
#elif SLANG_PROCESSOR_X86_64 && SLANG_VC
    return __popcnt16(v);
#else
    uint32_t c = 0;
    while (v)
    {
        c++;
        v &= v - 1;
    }
    return c;
#endif
}

SLANG_FORCE_INLINE half U16_ashalf(uint16_t x)
{
    Union16 u;
    u.u = x;
    return u.h;
}

// ----------------------------- I16 -----------------------------------------
SLANG_FORCE_INLINE uint32_t I16_countbits(int16_t v)
{
    return U16_countbits(uint16_t(v));
}

// ----------------------------- U8 -----------------------------------------
SLANG_FORCE_INLINE uint32_t U8_countbits(uint8_t v)
{
    // No native 8bit __popcnt yet, just cast and use 16bit variant
    return U16_countbits(uint16_t(v));
}

// ----------------------------- I8 -----------------------------------------
SLANG_FORCE_INLINE uint32_t I8_countbits(int16_t v)
{
    return U8_countbits(uint8_t(v));
}

// ----------------------------- U32 -----------------------------------------

SLANG_FORCE_INLINE uint32_t U32_abs(uint32_t f)
{
    return f;
}

SLANG_FORCE_INLINE uint32_t U32_min(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE uint32_t U32_max(uint32_t a, uint32_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE float U32_asfloat(uint32_t x)
{
    Union32 u;
    u.u = x;
    return u.f;
}
SLANG_FORCE_INLINE uint32_t U32_asint(int32_t x)
{
    return uint32_t(x);
}

SLANG_FORCE_INLINE double U32_asdouble(uint32_t low, uint32_t hi)
{
    Union64 u;
    u.u = (uint64_t(hi) << 32) | low;
    return u.d;
}

SLANG_FORCE_INLINE uint32_t U32_countbits(uint32_t v)
{
#if SLANG_GCC_FAMILY && !defined(SLANG_LLVM)
    return __builtin_popcount(v);
#elif SLANG_PROCESSOR_X86_64 && SLANG_VC
    return __popcnt(v);
#else
    uint32_t c = 0;
    while (v)
    {
        c++;
        v &= v - 1;
    }
    return c;
#endif
}

SLANG_FORCE_INLINE uint32_t U32_firstbitlow(uint32_t v)
{
    if (v == 0)
        return ~0u;

#if SLANG_GCC_FAMILY && !defined(SLANG_LLVM)
    // __builtin_ctz returns number of trailing zeros, which is the 0-based index of first set bit
    return __builtin_ctz(v);
#elif SLANG_PROCESSOR_X86_64 && SLANG_VC
    // _BitScanForward returns 1 on success, 0 on failure, and sets index
    unsigned long index;
    return _BitScanForward(&index, v) ? index : ~0u;
#else
    // Generic implementation - find first set bit
    uint32_t result = 0;
    while (result < 32 && !(v & (1u << result)))
        result++;
    return result;
#endif
}

SLANG_FORCE_INLINE uint32_t U32_firstbithigh(uint32_t v)
{
    if (v == 0)
        return ~0u;
#if SLANG_GCC_FAMILY && !defined(SLANG_LLVM)
    // __builtin_clz returns number of leading zeros
    // firstbithigh should return 0-based bit position of MSB
    return 31 - __builtin_clz(v);
#elif SLANG_PROCESSOR_X86_64 && SLANG_VC
    // _BitScanReverse returns 1 on success, 0 on failure, and sets index
    unsigned long index;
    return _BitScanReverse(&index, v) ? index : ~0u;
#else
    // Generic implementation - find highest set bit
    int result = 31;
    while (result >= 0 && !(v & (1u << result)))
        result--;
    return result;
#endif
}

SLANG_FORCE_INLINE uint32_t U32_reversebits(uint32_t v)
{
    v = ((v >> 1) & 0x55555555u) | ((v & 0x55555555u) << 1);
    v = ((v >> 2) & 0x33333333u) | ((v & 0x33333333u) << 2);
    v = ((v >> 4) & 0x0F0F0F0Fu) | ((v & 0x0F0F0F0Fu) << 4);
    v = ((v >> 8) & 0x00FF00FFu) | ((v & 0x00FF00FFu) << 8);
    v = (v >> 16) | (v << 16);
    return v;
}

// ----------------------------- I32 -----------------------------------------

SLANG_FORCE_INLINE int32_t I32_abs(int32_t f)
{
    return (f < 0) ? -f : f;
}

SLANG_FORCE_INLINE int32_t I32_min(int32_t a, int32_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE int32_t I32_max(int32_t a, int32_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE float I32_asfloat(int32_t x)
{
    Union32 u;
    u.i = x;
    return u.f;
}
SLANG_FORCE_INLINE uint32_t I32_asuint(int32_t x)
{
    return uint32_t(x);
}
SLANG_FORCE_INLINE double I32_asdouble(int32_t low, int32_t hi)
{
    Union64 u;
    u.u = (uint64_t(hi) << 32) | uint32_t(low);
    return u.d;
}

SLANG_FORCE_INLINE uint32_t I32_countbits(int32_t v)
{
    return U32_countbits(uint32_t(v));
}

SLANG_FORCE_INLINE uint32_t I32_firstbitlow(int32_t v)
{
    return U32_firstbitlow(uint32_t(v));
}

SLANG_FORCE_INLINE uint32_t I32_firstbithigh(int32_t v)
{
    if (v < 0)
        v = ~v;
    return U32_firstbithigh(uint32_t(v));
}

SLANG_FORCE_INLINE int32_t I32_reversebits(int32_t v)
{
    return U32_reversebits(int32_t(v));
}

// ----------------------------- U64 -----------------------------------------

SLANG_FORCE_INLINE uint64_t U64_abs(uint64_t f)
{
    return f;
}

SLANG_FORCE_INLINE uint64_t U64_min(uint64_t a, uint64_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE uint64_t U64_max(uint64_t a, uint64_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE uint32_t U64_countbits(uint64_t v)
{
#if SLANG_GCC_FAMILY && !defined(SLANG_LLVM)
    return uint32_t(__builtin_popcountll(v));
#elif SLANG_PROCESSOR_X86_64 && SLANG_VC
    return uint32_t(__popcnt64(v));
#else
    uint32_t c = 0;
    while (v)
    {
        c++;
        v &= v - 1;
    }
    return c;
#endif
}

SLANG_FORCE_INLINE uint32_t U64_firstbitlow(uint64_t v)
{
    if (v == 0)
        return ~uint32_t(0);

#if SLANG_GCC_FAMILY && !defined(SLANG_LLVM)
    // __builtin_ctz returns number of trailing zeros, which is the 0-based index of first set bit
    return __builtin_ctz(v);
#elif SLANG_PROCESSOR_X86_64 && SLANG_VC
    // _BitScanForward returns 1 on success, 0 on failure, and sets index
    unsigned long index;
    return _BitScanForward64(&index, v) ? index : ~uint32_t(0);
#else
    // Generic implementation - find first set bit
    uint32_t result = 0;
    while (result < 64 && !(v & (uint64_t(1) << result)))
        result++;
    return result;
#endif
}

SLANG_FORCE_INLINE uint32_t U64_firstbithigh(uint64_t v)
{
    if (v == 0)
        return ~uint32_t(0);

#if SLANG_GCC_FAMILY && !defined(SLANG_LLVM)
    // __builtin_clz returns number of leading zeros
    // firstbithigh should return 0-based bit position of MSB
    return 63 - __builtin_clz(v);
#elif SLANG_PROCESSOR_X86_64 && SLANG_VC
    // _BitScanReverse returns 1 on success, 0 on failure, and sets index
    unsigned long index;
    return _BitScanReverse64(&index, v) ? index : ~uint32_t(0);
#else
    // Generic implementation - find highest set bit
    int result = 63;
    while (result >= 0 && !(v & (uint64_t(1) << result)))
        result--;
    return result;
#endif
}

SLANG_FORCE_INLINE uint64_t U64_reversebits(uint64_t v)
{
    v = ((v >> 1) & 0x5555555555555555ull) | ((v & 0x5555555555555555ull) << 1);
    v = ((v >> 2) & 0x3333333333333333ull) | ((v & 0x3333333333333333ull) << 2);
    v = ((v >> 4) & 0x0F0F0F0F0F0F0F0Full) | ((v & 0x0F0F0F0F0F0F0F0Full) << 4);
    v = ((v >> 8) & 0x00FF00FF00FF00FFull) | ((v & 0x00FF00FF00FF00FFull) << 8);
    v = ((v >> 16) & 0x0000FFFF0000FFFFull) | ((v & 0x0000FFFF0000FFFFull) << 16);
    v = (v >> 32) | (v << 32);
    return v;
}

// ----------------------------- I64 -----------------------------------------

SLANG_FORCE_INLINE int64_t I64_abs(int64_t f)
{
    return (f < 0) ? -f : f;
}

SLANG_FORCE_INLINE int64_t I64_min(int64_t a, int64_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE int64_t I64_max(int64_t a, int64_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE uint32_t I64_countbits(int64_t v)
{
    return U64_countbits(uint64_t(v));
}

SLANG_FORCE_INLINE uint32_t I64_firstbitlow(int64_t v)
{
    return U64_firstbitlow(uint64_t(v));
}

SLANG_FORCE_INLINE uint32_t I64_firstbithigh(int64_t v)
{
    if (v < 0)
        v = ~v;
    return U64_firstbithigh(uint64_t(v));
}

SLANG_FORCE_INLINE int64_t I64_reversebits(int64_t v)
{
    return int64_t(U64_reversebits(uint64_t(v)));
}

// ----------------------------- UPTR -----------------------------------------

SLANG_FORCE_INLINE uintptr_t UPTR_abs(uintptr_t f)
{
    return f;
}

SLANG_FORCE_INLINE uintptr_t UPTR_min(uintptr_t a, uintptr_t b)
{
    return a < b ? a : b;
}

SLANG_FORCE_INLINE uintptr_t UPTR_max(uintptr_t a, uintptr_t b)
{
    return a > b ? a : b;
}

// ----------------------------- IPTR -----------------------------------------

SLANG_FORCE_INLINE intptr_t IPTR_abs(intptr_t f)
{
    return (f < 0) ? -f : f;
}

SLANG_FORCE_INLINE intptr_t IPTR_min(intptr_t a, intptr_t b)
{
    return a < b ? a : b;
}

SLANG_FORCE_INLINE intptr_t IPTR_max(intptr_t a, intptr_t b)
{
    return a > b ? a : b;
}

// ----------------------------- Interlocked ---------------------------------

#if SLANG_LLVM

#else // SLANG_LLVM

#ifdef _WIN32
#include <intrin.h>
#endif

SLANG_FORCE_INLINE void InterlockedAdd(uint32_t* dest, uint32_t value, uint32_t* oldValue)
{
#ifdef _WIN32
    *oldValue = _InterlockedExchangeAdd((long*)dest, (long)value);
#else
    *oldValue = __sync_fetch_and_add(dest, value);
#endif
}

#endif // SLANG_LLVM


// ----------------------- fmod --------------------------
SLANG_FORCE_INLINE float _slang_fmod(float x, float y)
{
    return F32_fmod(x, y);
}
SLANG_FORCE_INLINE double _slang_fmod(double x, double y)
{
    return F64_fmod(x, y);
}

#ifdef SLANG_PRELUDE_NAMESPACE
}
#endif

#endif

#ifndef SLANG_PRELUDE_CPP_TYPES_H
#define SLANG_PRELUDE_CPP_TYPES_H

#ifdef SLANG_PRELUDE_NAMESPACE
namespace SLANG_PRELUDE_NAMESPACE
{
#endif

#ifndef SLANG_FORCE_INLINE
#define SLANG_FORCE_INLINE inline
#endif

#ifndef SLANG_PRELUDE_CPP_TYPES_CORE_H
#define SLANG_PRELUDE_CPP_TYPES_CORE_H

#ifndef SLANG_PRELUDE_ASSERT
#ifdef SLANG_PRELUDE_ENABLE_ASSERT
#define SLANG_PRELUDE_ASSERT(VALUE) assert(VALUE)
#else
#define SLANG_PRELUDE_ASSERT(VALUE)
#endif
#endif

// Since we are using unsigned arithmatic care is need in this comparison.
// It is *assumed* that sizeInBytes >= elemSize. Which means (sizeInBytes >= elemSize) >= 0
// Which means only a single test is needed

// Asserts for bounds checking.
// It is assumed index/count are unsigned types.
#define SLANG_BOUND_ASSERT(index, count) SLANG_PRELUDE_ASSERT(index < count);
#define SLANG_BOUND_ASSERT_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_PRELUDE_ASSERT(index <= (sizeInBytes - elemSize) && (index & 3) == 0);

// Macros to zero index if an access is out of range
#define SLANG_BOUND_ZERO_INDEX(index, count) index = (index < count) ? index : 0;
#define SLANG_BOUND_ZERO_INDEX_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    index = (index <= (sizeInBytes - elemSize)) ? index : 0;

// The 'FIX' macro define how the index is fixed. The default is to do nothing. If
// SLANG_ENABLE_BOUND_ZERO_INDEX the fix macro will zero the index, if out of range
#ifdef SLANG_ENABLE_BOUND_ZERO_INDEX
#define SLANG_BOUND_FIX(index, count) SLANG_BOUND_ZERO_INDEX(index, count)
#define SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_BOUND_ZERO_INDEX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#define SLANG_BOUND_FIX_FIXED_ARRAY(index, count) SLANG_BOUND_ZERO_INDEX(index, count)
#else
#define SLANG_BOUND_FIX(index, count)
#define SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#define SLANG_BOUND_FIX_FIXED_ARRAY(index, count)
#endif

#ifndef SLANG_BOUND_CHECK
#define SLANG_BOUND_CHECK(index, count) \
    SLANG_BOUND_ASSERT(index, count) SLANG_BOUND_FIX(index, count)
#endif

#ifndef SLANG_BOUND_CHECK_BYTE_ADDRESS
#define SLANG_BOUND_CHECK_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_BOUND_ASSERT_BYTE_ADDRESS(index, elemSize, sizeInBytes)    \
    SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#endif

#ifndef SLANG_BOUND_CHECK_FIXED_ARRAY
#define SLANG_BOUND_CHECK_FIXED_ARRAY(index, count) \
    SLANG_BOUND_ASSERT(index, count) SLANG_BOUND_FIX_FIXED_ARRAY(index, count)
#endif

struct TypeInfo
{
    size_t typeSize;
};

template<typename T, size_t SIZE>
struct FixedArray
{
    const T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK_FIXED_ARRAY(index, SIZE);
        return m_data[index];
    }
    T& operator[](size_t index)
    {
        SLANG_BOUND_CHECK_FIXED_ARRAY(index, SIZE);
        return m_data[index];
    }

    T m_data[SIZE];
};

// An array that has no specified size, becomes a 'Array'. This stores the size so it can
// potentially do bounds checking.
template<typename T>
struct Array
{
    const T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    T& operator[](size_t index)
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }

    T* data;
    size_t count;
};

/* Constant buffers become a pointer to the contained type, so ConstantBuffer<T> becomes T* in C++
 * code.
 */

template<typename T, int COUNT>
struct Vector;

template<typename T>
struct Vector<T, 1>
{
    T x;
    const T& operator[](size_t /*index*/) const { return x; }
    T& operator[](size_t /*index*/) { return x; }
    operator T() const { return x; }
    Vector() = default;
    Vector(T scalar) { x = scalar; }
    template<typename U>
    Vector(Vector<U, 1> other)
    {
        x = (T)other.x;
    }
    template<typename U, int otherSize>
    Vector(Vector<U, otherSize> other)
    {
        int minSize = 1;
        if (otherSize < minSize)
            minSize = otherSize;
        for (int i = 0; i < minSize; i++)
            (*this)[i] = (T)other[i];
    }
};

template<typename T>
struct Vector<T, 2>
{
    T x, y;
    const T& operator[](size_t index) const { return index == 0 ? x : y; }
    T& operator[](size_t index) { return index == 0 ? x : y; }
    Vector() = default;
    Vector(T scalar) { x = y = scalar; }
    Vector(T _x, T _y)
    {
        x = _x;
        y = _y;
    }
    template<typename U>
    Vector(Vector<U, 2> other)
    {
        x = (T)other.x;
        y = (T)other.y;
    }
    template<typename U, int otherSize>
    Vector(Vector<U, otherSize> other)
    {
        int minSize = 2;
        if (otherSize < minSize)
            minSize = otherSize;
        for (int i = 0; i < minSize; i++)
            (*this)[i] = (T)other[i];
    }
};

template<typename T>
struct Vector<T, 3>
{
    T x, y, z;
    const T& operator[](size_t index) const { return *((T*)(this) + index); }
    T& operator[](size_t index) { return *((T*)(this) + index); }

    Vector() = default;
    Vector(T scalar) { x = y = z = scalar; }
    Vector(T _x, T _y, T _z)
    {
        x = _x;
        y = _y;
        z = _z;
    }
    template<typename U>
    Vector(Vector<U, 3> other)
    {
        x = (T)other.x;
        y = (T)other.y;
        z = (T)other.z;
    }
    template<typename U, int otherSize>
    Vector(Vector<U, otherSize> other)
    {
        int minSize = 3;
        if (otherSize < minSize)
            minSize = otherSize;
        for (int i = 0; i < minSize; i++)
            (*this)[i] = (T)other[i];
    }
};

template<typename T>
struct Vector<T, 4>
{
    T x, y, z, w;

    const T& operator[](size_t index) const { return *((T*)(this) + index); }
    T& operator[](size_t index) { return *((T*)(this) + index); }
    Vector() = default;
    Vector(T scalar) { x = y = z = w = scalar; }
    Vector(T _x, T _y, T _z, T _w)
    {
        x = _x;
        y = _y;
        z = _z;
        w = _w;
    }
    template<typename U, int otherSize>
    Vector(Vector<U, otherSize> other)
    {
        int minSize = 4;
        if (otherSize < minSize)
            minSize = otherSize;
        for (int i = 0; i < minSize; i++)
            (*this)[i] = (T)other[i];
    }
};

template<typename T, int N>
SLANG_FORCE_INLINE Vector<T, N> _slang_select(
    Vector<bool, N> condition,
    Vector<T, N> v0,
    Vector<T, N> v1)
{
    Vector<T, N> result;
    for (int i = 0; i < N; i++)
    {
        result[i] = condition[i] ? v0[i] : v1[i];
    }
    return result;
}

template<typename T>
SLANG_FORCE_INLINE T _slang_select(bool condition, T v0, T v1)
{
    return condition ? v0 : v1;
}

template<typename T, int N>
SLANG_FORCE_INLINE T _slang_vector_get_element(Vector<T, N> x, int index)
{
    return x[index];
}

template<typename T, int N>
SLANG_FORCE_INLINE const T* _slang_vector_get_element_ptr(const Vector<T, N>* x, int index)
{
    return &((*const_cast<Vector<T, N>*>(x))[index]);
}

template<typename T, int N>
SLANG_FORCE_INLINE T* _slang_vector_get_element_ptr(Vector<T, N>* x, int index)
{
    return &((*x)[index]);
}

template<typename T, int n, typename OtherT, int m>
SLANG_FORCE_INLINE Vector<T, n> _slang_vector_reshape(const Vector<OtherT, m> other)
{
    Vector<T, n> result;
    for (int i = 0; i < n; i++)
    {
        OtherT otherElement = T(0);
        if (i < m)
            otherElement = _slang_vector_get_element(other, i);
        *_slang_vector_get_element_ptr(&result, i) = (T)otherElement;
    }
    return result;
}

typedef uint32_t uint;

#define SLANG_VECTOR_BINARY_OP(T, op)            \
    template<int n>                              \
    SLANG_FORCE_INLINE Vector<T, n> operator op( \
        const Vector<T, n>& thisVal,             \
        const Vector<T, n>& other)               \
    {                                            \
        Vector<T, n> result;                     \
        for (int i = 0; i < n; i++)              \
            result[i] = thisVal[i] op other[i];  \
        return result;                           \
    }
#define SLANG_VECTOR_BINARY_COMPARE_OP(T, op)       \
    template<int n>                                 \
    SLANG_FORCE_INLINE Vector<bool, n> operator op( \
        const Vector<T, n>& thisVal,                \
        const Vector<T, n>& other)                  \
    {                                               \
        Vector<bool, n> result;                     \
        for (int i = 0; i < n; i++)                 \
            result[i] = thisVal[i] op other[i];     \
        return result;                              \
    }

#define SLANG_VECTOR_UNARY_OP(T, op)                                         \
    template<int n>                                                          \
    SLANG_FORCE_INLINE Vector<T, n> operator op(const Vector<T, n>& thisVal) \
    {                                                                        \
        Vector<T, n> result;                                                 \
        for (int i = 0; i < n; i++)                                          \
            result[i] = op thisVal[i];                                       \
        return result;                                                       \
    }
#define SLANG_INT_VECTOR_OPS(T)           \
    SLANG_VECTOR_BINARY_OP(T, +)          \
    SLANG_VECTOR_BINARY_OP(T, -)          \
    SLANG_VECTOR_BINARY_OP(T, *)          \
    SLANG_VECTOR_BINARY_OP(T, /)          \
    SLANG_VECTOR_BINARY_OP(T, &)          \
    SLANG_VECTOR_BINARY_OP(T, |)          \
    SLANG_VECTOR_BINARY_OP(T, &&)         \
    SLANG_VECTOR_BINARY_OP(T, ||)         \
    SLANG_VECTOR_BINARY_OP(T, ^)          \
    SLANG_VECTOR_BINARY_OP(T, %)          \
    SLANG_VECTOR_BINARY_OP(T, >>)         \
    SLANG_VECTOR_BINARY_OP(T, <<)         \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, >)  \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, <)  \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, >=) \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, <=) \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, ==) \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, !=) \
    SLANG_VECTOR_UNARY_OP(T, !)           \
    SLANG_VECTOR_UNARY_OP(T, ~)
#define SLANG_FLOAT_VECTOR_OPS(T)         \
    SLANG_VECTOR_BINARY_OP(T, +)          \
    SLANG_VECTOR_BINARY_OP(T, -)          \
    SLANG_VECTOR_BINARY_OP(T, *)          \
    SLANG_VECTOR_BINARY_OP(T, /)          \
    SLANG_VECTOR_UNARY_OP(T, -)           \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, >)  \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, <)  \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, >=) \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, <=) \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, ==) \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, !=)

SLANG_INT_VECTOR_OPS(bool)
SLANG_INT_VECTOR_OPS(int)
SLANG_INT_VECTOR_OPS(int8_t)
SLANG_INT_VECTOR_OPS(int16_t)
SLANG_INT_VECTOR_OPS(int64_t)
SLANG_INT_VECTOR_OPS(uint)
SLANG_INT_VECTOR_OPS(uint8_t)
SLANG_INT_VECTOR_OPS(uint16_t)
SLANG_INT_VECTOR_OPS(uint64_t)
#if SLANG_INTPTR_TYPE_IS_DISTINCT
SLANG_INT_VECTOR_OPS(intptr_t)
SLANG_INT_VECTOR_OPS(uintptr_t)
#endif

SLANG_FLOAT_VECTOR_OPS(float)
SLANG_FLOAT_VECTOR_OPS(double)

#define SLANG_VECTOR_INT_NEG_OP(T)                      \
    template<int N>                                     \
    Vector<T, N> operator-(const Vector<T, N>& thisVal) \
    {                                                   \
        Vector<T, N> result;                            \
        for (int i = 0; i < N; i++)                     \
            result[i] = 0 - thisVal[i];                 \
        return result;                                  \
    }
SLANG_VECTOR_INT_NEG_OP(int)
SLANG_VECTOR_INT_NEG_OP(int8_t)
SLANG_VECTOR_INT_NEG_OP(int16_t)
SLANG_VECTOR_INT_NEG_OP(int64_t)
SLANG_VECTOR_INT_NEG_OP(uint)
SLANG_VECTOR_INT_NEG_OP(uint8_t)
SLANG_VECTOR_INT_NEG_OP(uint16_t)
SLANG_VECTOR_INT_NEG_OP(uint64_t)
#if SLANG_INTPTR_TYPE_IS_DISTINCT
SLANG_VECTOR_INT_NEG_OP(intptr_t)
SLANG_VECTOR_INT_NEG_OP(uintptr_t)
#endif

#define SLANG_FLOAT_VECTOR_MOD(T)                                               \
    template<int N>                                                             \
    Vector<T, N> operator%(const Vector<T, N>& left, const Vector<T, N>& right) \
    {                                                                           \
        Vector<T, N> result;                                                    \
        for (int i = 0; i < N; i++)                                             \
            result[i] = _slang_fmod(left[i], right[i]);                         \
        return result;                                                          \
    }

SLANG_FLOAT_VECTOR_MOD(float)
SLANG_FLOAT_VECTOR_MOD(double)
#undef SLANG_FLOAT_VECTOR_MOD
#undef SLANG_VECTOR_BINARY_OP
#undef SLANG_VECTOR_UNARY_OP
#undef SLANG_INT_VECTOR_OPS
#undef SLANG_FLOAT_VECTOR_OPS
#undef SLANG_VECTOR_INT_NEG_OP
#undef SLANG_FLOAT_VECTOR_MOD

template<typename T, int ROWS, int COLS>
struct Matrix
{
    Vector<T, COLS> rows[ROWS];
    const Vector<T, COLS>& operator[](size_t index) const { return rows[index]; }
    Vector<T, COLS>& operator[](size_t index) { return rows[index]; }
    Matrix() = default;
    Matrix(T scalar)
    {
        for (int i = 0; i < ROWS; i++)
            rows[i] = Vector<T, COLS>(scalar);
    }
    Matrix(const Vector<T, COLS>& row0) { rows[0] = row0; }
    Matrix(const Vector<T, COLS>& row0, const Vector<T, COLS>& row1)
    {
        rows[0] = row0;
        rows[1] = row1;
    }
    Matrix(const Vector<T, COLS>& row0, const Vector<T, COLS>& row1, const Vector<T, COLS>& row2)
    {
        rows[0] = row0;
        rows[1] = row1;
        rows[2] = row2;
    }
    Matrix(
        const Vector<T, COLS>& row0,
        const Vector<T, COLS>& row1,
        const Vector<T, COLS>& row2,
        const Vector<T, COLS>& row3)
    {
        rows[0] = row0;
        rows[1] = row1;
        rows[2] = row2;
        rows[3] = row3;
    }
    template<typename U, int otherRow, int otherCol>
    Matrix(const Matrix<U, otherRow, otherCol>& other)
    {
        int minRow = ROWS;
        int minCol = COLS;
        if (minRow > otherRow)
            minRow = otherRow;
        if (minCol > otherCol)
            minCol = otherCol;
        for (int i = 0; i < minRow; i++)
            for (int j = 0; j < minCol; j++)
                rows[i][j] = (T)other.rows[i][j];
    }
    Matrix(T v0, T v1, T v2, T v3)
    {
        rows[0][0] = v0;
        rows[0][1] = v1;
        rows[1][0] = v2;
        rows[1][1] = v3;
    }
    Matrix(T v0, T v1, T v2, T v3, T v4, T v5)
    {
        if (COLS == 3)
        {
            rows[0][0] = v0;
            rows[0][1] = v1;
            rows[0][2] = v2;
            rows[1][0] = v3;
            rows[1][1] = v4;
            rows[1][2] = v5;
        }
        else
        {
            rows[0][0] = v0;
            rows[0][1] = v1;
            rows[1][0] = v2;
            rows[1][1] = v3;
            rows[2][0] = v4;
            rows[2][1] = v5;
        }
    }
    Matrix(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7)
    {
        if (COLS == 4)
        {
            rows[0][0] = v0;
            rows[0][1] = v1;
            rows[0][2] = v2;
            rows[0][3] = v3;
            rows[1][0] = v4;
            rows[1][1] = v5;
            rows[1][2] = v6;
            rows[1][3] = v7;
        }
        else
        {
            rows[0][0] = v0;
            rows[0][1] = v1;
            rows[1][0] = v2;
            rows[1][1] = v3;
            rows[2][0] = v4;
            rows[2][1] = v5;
            rows[3][0] = v6;
            rows[3][1] = v7;
        }
    }
    Matrix(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8)
    {
        rows[0][0] = v0;
        rows[0][1] = v1;
        rows[0][2] = v2;
        rows[1][0] = v3;
        rows[1][1] = v4;
        rows[1][2] = v5;
        rows[2][0] = v6;
        rows[2][1] = v7;
        rows[2][2] = v8;
    }
    Matrix(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8, T v9, T v10, T v11)
    {
        if (COLS == 4)
        {
            rows[0][0] = v0;
            rows[0][1] = v1;
            rows[0][2] = v2;
            rows[0][3] = v3;
            rows[1][0] = v4;
            rows[1][1] = v5;
            rows[1][2] = v6;
            rows[1][3] = v7;
            rows[2][0] = v8;
            rows[2][1] = v9;
            rows[2][2] = v10;
            rows[2][3] = v11;
        }
        else
        {
            rows[0][0] = v0;
            rows[0][1] = v1;
            rows[0][2] = v2;
            rows[1][0] = v3;
            rows[1][1] = v4;
            rows[1][2] = v5;
            rows[2][0] = v6;
            rows[2][1] = v7;
            rows[2][2] = v8;
            rows[3][0] = v9;
            rows[3][1] = v10;
            rows[3][2] = v11;
        }
    }
    Matrix(
        T v0,
        T v1,
        T v2,
        T v3,
        T v4,
        T v5,
        T v6,
        T v7,
        T v8,
        T v9,
        T v10,
        T v11,
        T v12,
        T v13,
        T v14,
        T v15)
    {
        rows[0][0] = v0;
        rows[0][1] = v1;
        rows[0][2] = v2;
        rows[0][3] = v3;
        rows[1][0] = v4;
        rows[1][1] = v5;
        rows[1][2] = v6;
        rows[1][3] = v7;
        rows[2][0] = v8;
        rows[2][1] = v9;
        rows[2][2] = v10;
        rows[2][3] = v11;
        rows[3][0] = v12;
        rows[3][1] = v13;
        rows[3][2] = v14;
        rows[3][3] = v15;
    }
};

#define SLANG_MATRIX_BINARY_OP(T, op)                                                         \
    template<int R, int C>                                                                    \
    Matrix<T, R, C> operator op(const Matrix<T, R, C>& thisVal, const Matrix<T, R, C>& other) \
    {                                                                                         \
        Matrix<T, R, C> result;                                                               \
        for (int i = 0; i < R; i++)                                                           \
            for (int j = 0; j < C; j++)                                                       \
                result.rows[i][j] = thisVal.rows[i][j] op other.rows[i][j];                   \
        return result;                                                                        \
    }

#define SLANG_MATRIX_BINARY_COMPARE_OP(T, op)                                                    \
    template<int R, int C>                                                                       \
    Matrix<bool, R, C> operator op(const Matrix<T, R, C>& thisVal, const Matrix<T, R, C>& other) \
    {                                                                                            \
        Matrix<bool, R, C> result;                                                               \
        for (int i = 0; i < R; i++)                                                              \
            for (int j = 0; j < C; j++)                                                          \
                result.rows[i][j] = thisVal.rows[i][j] op other.rows[i][j];                      \
        return result;                                                                           \
    }

#define SLANG_MATRIX_UNARY_OP(T, op)                            \
    template<int R, int C>                                      \
    Matrix<T, R, C> operator op(const Matrix<T, R, C>& thisVal) \
    {                                                           \
        Matrix<T, R, C> result;                                 \
        for (int i = 0; i < R; i++)                             \
            for (int j = 0; j < C; j++)                         \
                result[i].rows[i][j] = op thisVal.rows[i][j];   \
        return result;                                          \
    }

#define SLANG_INT_MATRIX_OPS(T)           \
    SLANG_MATRIX_BINARY_OP(T, +)          \
    SLANG_MATRIX_BINARY_OP(T, -)          \
    SLANG_MATRIX_BINARY_OP(T, *)          \
    SLANG_MATRIX_BINARY_OP(T, /)          \
    SLANG_MATRIX_BINARY_OP(T, &)          \
    SLANG_MATRIX_BINARY_OP(T, |)          \
    SLANG_MATRIX_BINARY_OP(T, &&)         \
    SLANG_MATRIX_BINARY_OP(T, ||)         \
    SLANG_MATRIX_BINARY_OP(T, ^)          \
    SLANG_MATRIX_BINARY_OP(T, %)          \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, >)  \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, <)  \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, >=) \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, <=) \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, ==) \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, !=) \
    SLANG_MATRIX_UNARY_OP(T, !)           \
    SLANG_MATRIX_UNARY_OP(T, ~)
#define SLANG_FLOAT_MATRIX_OPS(T)         \
    SLANG_MATRIX_BINARY_OP(T, +)          \
    SLANG_MATRIX_BINARY_OP(T, -)          \
    SLANG_MATRIX_BINARY_OP(T, *)          \
    SLANG_MATRIX_BINARY_OP(T, /)          \
    SLANG_MATRIX_UNARY_OP(T, -)           \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, >)  \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, <)  \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, >=) \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, <=) \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, ==) \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, !=)
SLANG_INT_MATRIX_OPS(int)
SLANG_INT_MATRIX_OPS(int8_t)
SLANG_INT_MATRIX_OPS(int16_t)
SLANG_INT_MATRIX_OPS(int64_t)
SLANG_INT_MATRIX_OPS(uint)
SLANG_INT_MATRIX_OPS(uint8_t)
SLANG_INT_MATRIX_OPS(uint16_t)
SLANG_INT_MATRIX_OPS(uint64_t)
#if SLANG_INTPTR_TYPE_IS_DISTINCT
SLANG_INT_MATRIX_OPS(intptr_t)
SLANG_INT_MATRIX_OPS(uintptr_t)
#endif

SLANG_FLOAT_MATRIX_OPS(float)
SLANG_FLOAT_MATRIX_OPS(double)

#define SLANG_MATRIX_INT_NEG_OP(T)                                        \
    template<int R, int C>                                                \
    SLANG_FORCE_INLINE Matrix<T, R, C> operator-(Matrix<T, R, C> thisVal) \
    {                                                                     \
        Matrix<T, R, C> result;                                           \
        for (int i = 0; i < R; i++)                                       \
            for (int j = 0; j < C; j++)                                   \
                result.rows[i][j] = 0 - thisVal.rows[i][j];               \
        return result;                                                    \
    }
SLANG_MATRIX_INT_NEG_OP(int)
SLANG_MATRIX_INT_NEG_OP(int8_t)
SLANG_MATRIX_INT_NEG_OP(int16_t)
SLANG_MATRIX_INT_NEG_OP(int64_t)
SLANG_MATRIX_INT_NEG_OP(uint)
SLANG_MATRIX_INT_NEG_OP(uint8_t)
SLANG_MATRIX_INT_NEG_OP(uint16_t)
SLANG_MATRIX_INT_NEG_OP(uint64_t)
#if SLANG_INTPTR_TYPE_IS_DISTINCT
SLANG_MATRIX_INT_NEG_OP(intptr_t)
SLANG_MATRIX_INT_NEG_OP(uintptr_t)
#endif

#define SLANG_FLOAT_MATRIX_MOD(T)                                                             \
    template<int R, int C>                                                                    \
    SLANG_FORCE_INLINE Matrix<T, R, C> operator%(Matrix<T, R, C> left, Matrix<T, R, C> right) \
    {                                                                                         \
        Matrix<T, R, C> result;                                                               \
        for (int i = 0; i < R; i++)                                                           \
            for (int j = 0; j < C; j++)                                                       \
                result.rows[i][j] = _slang_fmod(left.rows[i][j], right.rows[i][j]);           \
        return result;                                                                        \
    }

SLANG_FLOAT_MATRIX_MOD(float)
SLANG_FLOAT_MATRIX_MOD(double)
#undef SLANG_FLOAT_MATRIX_MOD
#undef SLANG_MATRIX_BINARY_OP
#undef SLANG_MATRIX_UNARY_OP
#undef SLANG_INT_MATRIX_OPS
#undef SLANG_FLOAT_MATRIX_OPS
#undef SLANG_MATRIX_INT_NEG_OP
#undef SLANG_FLOAT_MATRIX_MOD

template<typename TResult, typename TInput>
TResult slang_bit_cast(TInput val)
{
    return *(TResult*)(&val);
}

#endif


typedef Vector<float, 2> float2;
typedef Vector<float, 3> float3;
typedef Vector<float, 4> float4;

typedef Vector<int32_t, 2> int2;
typedef Vector<int32_t, 3> int3;
typedef Vector<int32_t, 4> int4;

typedef Vector<uint32_t, 2> uint2;
typedef Vector<uint32_t, 3> uint3;
typedef Vector<uint32_t, 4> uint4;

// We can just map `NonUniformResourceIndex` type directly to the index type on CPU, as CPU does not
// require any special handling around such accesses.
typedef size_t NonUniformResourceIndex;

// ----------------------------- ResourceType -----------------------------------------

// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-structuredbuffer-getdimensions
// Missing  Load(_In_  int  Location, _Out_ uint Status);

template<typename T>
struct RWStructuredBuffer
{
    SLANG_FORCE_INLINE T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    const T& Load(size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    void GetDimensions(uint32_t* outNumStructs, uint32_t* outStride)
    {
        *outNumStructs = uint32_t(count);
        *outStride = uint32_t(sizeof(T));
    }

    T* data;
    size_t count;
};

template<typename T>
struct StructuredBuffer
{
    SLANG_FORCE_INLINE T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    T& Load(size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    void GetDimensions(uint32_t* outNumStructs, uint32_t* outStride)
    {
        *outNumStructs = uint32_t(count);
        *outStride = uint32_t(sizeof(T));
    }

    T* data;
    size_t count;
};


template<typename T>
struct RWBuffer
{
    SLANG_FORCE_INLINE T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    const T& Load(size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    void GetDimensions(uint32_t* outCount) { *outCount = uint32_t(count); }

    T* data;
    size_t count;
};

template<typename T>
struct Buffer
{
    SLANG_FORCE_INLINE const T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    const T& Load(size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    void GetDimensions(uint32_t* outCount) { *outCount = uint32_t(count); }

    T* data;
    size_t count;
};

// Missing  Load(_In_  int  Location, _Out_ uint Status);
struct ByteAddressBuffer
{
    void GetDimensions(uint32_t* outDim) const { *outDim = uint32_t(sizeInBytes); }
    uint32_t Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        return data[index >> 2];
    }
    uint2 Load2(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint2{data[dataIdx], data[dataIdx + 1]};
    }
    uint3 Load3(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint3{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2]};
    }
    uint4 Load4(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint4{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2], data[dataIdx + 3]};
    }
    template<typename T>
    T Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        return *(const T*)(((const char*)data) + index);
    }

    const uint32_t* data;
    size_t sizeInBytes; //< Must be multiple of 4
};

// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-rwbyteaddressbuffer
// Missing support for Atomic operations
// Missing support for Load with status
struct RWByteAddressBuffer
{
    void GetDimensions(uint32_t* outDim) const { *outDim = uint32_t(sizeInBytes); }

    uint32_t Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        return data[index >> 2];
    }
    uint2 Load2(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint2{data[dataIdx], data[dataIdx + 1]};
    }
    uint3 Load3(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint3{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2]};
    }
    uint4 Load4(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint4{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2], data[dataIdx + 3]};
    }
    template<typename T>
    T Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        return *(const T*)(((const char*)data) + index);
    }

    void Store(size_t index, uint32_t v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        data[index >> 2] = v;
    }
    void Store2(size_t index, uint2 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
    }
    void Store3(size_t index, uint3 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
        data[dataIdx + 2] = v.z;
    }
    void Store4(size_t index, uint4 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
        data[dataIdx + 2] = v.z;
        data[dataIdx + 3] = v.w;
    }
    template<typename T>
    void Store(size_t index, T const& value) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        *(T*)(((char*)data) + index) = value;
    }

    uint32_t* data;
    size_t sizeInBytes; //< Must be multiple of 4
};

struct ISamplerState;
struct ISamplerComparisonState;

struct SamplerState
{
    ISamplerState* state;
};

struct SamplerComparisonState
{
    ISamplerComparisonState* state;
};

#ifndef SLANG_RESOURCE_SHAPE
#define SLANG_RESOURCE_SHAPE
typedef unsigned int SlangResourceShape;
enum
{
    SLANG_RESOURCE_BASE_SHAPE_MASK = 0x0F,

    SLANG_RESOURCE_NONE = 0x00,

    SLANG_TEXTURE_1D = 0x01,
    SLANG_TEXTURE_2D = 0x02,
    SLANG_TEXTURE_3D = 0x03,
    SLANG_TEXTURE_CUBE = 0x04,
    SLANG_TEXTURE_BUFFER = 0x05,

    SLANG_STRUCTURED_BUFFER = 0x06,
    SLANG_BYTE_ADDRESS_BUFFER = 0x07,
    SLANG_RESOURCE_UNKNOWN = 0x08,
    SLANG_ACCELERATION_STRUCTURE = 0x09,
    SLANG_TEXTURE_SUBPASS = 0x0A,

    SLANG_RESOURCE_EXT_SHAPE_MASK = 0xF0,

    SLANG_TEXTURE_FEEDBACK_FLAG = 0x10,
    SLANG_TEXTURE_ARRAY_FLAG = 0x40,
    SLANG_TEXTURE_MULTISAMPLE_FLAG = 0x80,

    SLANG_TEXTURE_1D_ARRAY = SLANG_TEXTURE_1D | SLANG_TEXTURE_ARRAY_FLAG,
    SLANG_TEXTURE_2D_ARRAY = SLANG_TEXTURE_2D | SLANG_TEXTURE_ARRAY_FLAG,
    SLANG_TEXTURE_CUBE_ARRAY = SLANG_TEXTURE_CUBE | SLANG_TEXTURE_ARRAY_FLAG,

    SLANG_TEXTURE_2D_MULTISAMPLE = SLANG_TEXTURE_2D | SLANG_TEXTURE_MULTISAMPLE_FLAG,
    SLANG_TEXTURE_2D_MULTISAMPLE_ARRAY =
        SLANG_TEXTURE_2D | SLANG_TEXTURE_MULTISAMPLE_FLAG | SLANG_TEXTURE_ARRAY_FLAG,
    SLANG_TEXTURE_SUBPASS_MULTISAMPLE = SLANG_TEXTURE_SUBPASS | SLANG_TEXTURE_MULTISAMPLE_FLAG,
};
#endif

//
struct TextureDimensions
{
    void reset()
    {
        shape = 0;
        width = height = depth = 0;
        numberOfLevels = 0;
        arrayElementCount = 0;
    }
    int getDimSizes(uint32_t outDims[4]) const
    {
        const auto baseShape = (shape & SLANG_RESOURCE_BASE_SHAPE_MASK);
        int count = 0;
        switch (baseShape)
        {
        case SLANG_TEXTURE_1D:
            {
                outDims[count++] = width;
                break;
            }
        case SLANG_TEXTURE_2D:
            {
                outDims[count++] = width;
                outDims[count++] = height;
                break;
            }
        case SLANG_TEXTURE_3D:
            {
                outDims[count++] = width;
                outDims[count++] = height;
                outDims[count++] = depth;
                break;
            }
        case SLANG_TEXTURE_CUBE:
            {
                outDims[count++] = width;
                outDims[count++] = height;
                outDims[count++] = 6;
                break;
            }
        }

        if (shape & SLANG_TEXTURE_ARRAY_FLAG)
        {
            outDims[count++] = arrayElementCount;
        }
        return count;
    }
    int getMIPDims(int outDims[3]) const
    {
        const auto baseShape = (shape & SLANG_RESOURCE_BASE_SHAPE_MASK);
        int count = 0;
        switch (baseShape)
        {
        case SLANG_TEXTURE_1D:
            {
                outDims[count++] = width;
                break;
            }
        case SLANG_TEXTURE_CUBE:
        case SLANG_TEXTURE_2D:
            {
                outDims[count++] = width;
                outDims[count++] = height;
                break;
            }
        case SLANG_TEXTURE_3D:
            {
                outDims[count++] = width;
                outDims[count++] = height;
                outDims[count++] = depth;
                break;
            }
        }
        return count;
    }
    int calcMaxMIPLevels() const
    {
        int dims[3];
        const int dimCount = getMIPDims(dims);
        for (int count = 1; true; count++)
        {
            bool allOne = true;
            for (int i = 0; i < dimCount; ++i)
            {
                if (dims[i] > 1)
                {
                    allOne = false;
                    dims[i] >>= 1;
                }
            }
            if (allOne)
            {
                return count;
            }
        }
    }

    uint32_t shape;
    uint32_t width, height, depth;
    uint32_t numberOfLevels;
    uint32_t arrayElementCount; ///< For array types, 0 otherwise
};


// Texture

struct ITexture
{
    virtual TextureDimensions GetDimensions(int mipLevel = -1) = 0;
    virtual void Load(const int32_t* v, void* outData, size_t dataSize) = 0;
    virtual void Sample(
        SamplerState samplerState,
        const float* loc,
        void* outData,
        size_t dataSize) = 0;
    virtual void SampleLevel(
        SamplerState samplerState,
        const float* loc,
        float level,
        void* outData,
        size_t dataSize) = 0;
};

template<typename T>
struct Texture1D
{
    void GetDimensions(uint32_t* outWidth) { *outWidth = texture->GetDimensions().width; }
    void GetDimensions(uint32_t mipLevel, uint32_t* outWidth, uint32_t* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    void GetDimensions(float* outWidth) { *outWidth = texture->GetDimensions().width; }
    void GetDimensions(uint32_t mipLevel, float* outWidth, float* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(const int2& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T Sample(SamplerState samplerState, float loc) const
    {
        T out;
        texture->Sample(samplerState, &loc, &out, sizeof(out));
        return out;
    }
    T SampleLevel(SamplerState samplerState, float loc, float level) const
    {
        T out;
        texture->SampleLevel(samplerState, &loc, level, &out, sizeof(out));
        return out;
    }

    ITexture* texture;
};

template<typename T>
struct Texture2D
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(const int3& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T Sample(SamplerState samplerState, const float2& loc) const
    {
        T out;
        texture->Sample(samplerState, &loc.x, &out, sizeof(out));
        return out;
    }
    T SampleLevel(SamplerState samplerState, const float2& loc, float level) const
    {
        T out;
        texture->SampleLevel(samplerState, &loc.x, level, &out, sizeof(out));
        return out;
    }

    ITexture* texture;
};

template<typename T>
struct Texture3D
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight, uint32_t* outDepth)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outDepth,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight, float* outDepth)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outDepth,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(const int4& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T Sample(SamplerState samplerState, const float3& loc) const
    {
        T out;
        texture->Sample(samplerState, &loc.x, &out, sizeof(out));
        return out;
    }
    T SampleLevel(SamplerState samplerState, const float3& loc, float level) const
    {
        T out;
        texture->SampleLevel(samplerState, &loc.x, level, &out, sizeof(out));
        return out;
    }

    ITexture* texture;
};

template<typename T>
struct TextureCube
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Sample(SamplerState samplerState, const float3& loc) const
    {
        T out;
        texture->Sample(samplerState, &loc.x, &out, sizeof(out));
        return out;
    }
    T SampleLevel(SamplerState samplerState, const float3& loc, float level) const
    {
        T out;
        texture->SampleLevel(samplerState, &loc.x, level, &out, sizeof(out));
        return out;
    }

    ITexture* texture;
};

template<typename T>
struct Texture1DArray
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outElements,
        uint32_t* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outNumberOfLevels = dims.numberOfLevels;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(float* outWidth, float* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outElements,
        float* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outNumberOfLevels = dims.numberOfLevels;
        *outElements = dims.arrayElementCount;
    }

    T Load(const int3& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T Sample(SamplerState samplerState, const float2& loc) const
    {
        T out;
        texture->Sample(samplerState, &loc.x, &out, sizeof(out));
        return out;
    }
    T SampleLevel(SamplerState samplerState, const float2& loc, float level) const
    {
        T out;
        texture->SampleLevel(samplerState, &loc.x, level, &out, sizeof(out));
        return out;
    }

    ITexture* texture;
};

template<typename T>
struct Texture2DArray
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight, uint32_t* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outElements,
        uint32_t* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    void GetDimensions(uint32_t* outWidth, float* outHeight, float* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outElements,
        float* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(const int4& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T Sample(SamplerState samplerState, const float3& loc) const
    {
        T out;
        texture->Sample(samplerState, &loc.x, &out, sizeof(out));
        return out;
    }
    T SampleLevel(SamplerState samplerState, const float3& loc, float level) const
    {
        T out;
        texture->SampleLevel(samplerState, &loc.x, level, &out, sizeof(out));
        return out;
    }

    ITexture* texture;
};

template<typename T>
struct TextureCubeArray
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight, uint32_t* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outElements,
        uint32_t* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    void GetDimensions(uint32_t* outWidth, float* outHeight, float* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outElements,
        float* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Sample(SamplerState samplerState, const float4& loc) const
    {
        T out;
        texture->Sample(samplerState, &loc.x, &out, sizeof(out));
        return out;
    }
    T SampleLevel(SamplerState samplerState, const float4& loc, float level) const
    {
        T out;
        texture->SampleLevel(samplerState, &loc.x, level, &out, sizeof(out));
        return out;
    }

    ITexture* texture;
};

/* !!!!!!!!!!!!!!!!!!!!!!!!!!! RWTexture !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

struct IRWTexture : ITexture
{
    /// Get the reference to the element at loc.
    virtual void* refAt(const uint32_t* loc) = 0;
};

template<typename T>
struct RWTexture1D
{
    void GetDimensions(uint32_t* outWidth) { *outWidth = texture->GetDimensions().width; }
    void GetDimensions(uint32_t mipLevel, uint32_t* outWidth, uint32_t* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    void GetDimensions(float* outWidth) { *outWidth = texture->GetDimensions().width; }
    void GetDimensions(uint32_t mipLevel, float* outWidth, float* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(int32_t loc) const
    {
        T out;
        texture->Load(&loc, &out, sizeof(out));
        return out;
    }
    T& operator[](uint32_t loc) { return *(T*)texture->refAt(&loc); }
    IRWTexture* texture;
};

template<typename T>
struct RWTexture2D
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(const int2& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T& operator[](const uint2& loc) { return *(T*)texture->refAt(&loc.x); }
    IRWTexture* texture;
};

template<typename T>
struct RWTexture3D
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight, uint32_t* outDepth)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outDepth,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight, float* outDepth)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outDepth,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(const int3& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T& operator[](const uint3& loc) { return *(T*)texture->refAt(&loc.x); }
    IRWTexture* texture;
};


template<typename T>
struct RWTexture1DArray
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outElements,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outElements,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(int2 loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T& operator[](uint2 loc) { return *(T*)texture->refAt(&loc.x); }

    IRWTexture* texture;
};

template<typename T>
struct RWTexture2DArray
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight, uint32_t* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outElements,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight, float* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outElements,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(const int3& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T& operator[](const uint3& loc) { return *(T*)texture->refAt(&loc.x); }

    IRWTexture* texture;
};

// FeedbackTexture

struct FeedbackType
{
};
struct SAMPLER_FEEDBACK_MIN_MIP : FeedbackType
{
};
struct SAMPLER_FEEDBACK_MIP_REGION_USED : FeedbackType
{
};

struct IFeedbackTexture
{
    virtual TextureDimensions GetDimensions(int mipLevel = -1) = 0;

    // Note here we pass the optional clamp parameter as a pointer. Passing nullptr means no clamp.
    // This was preferred over having two function definitions, and having to differentiate their
    // names
    virtual void WriteSamplerFeedback(
        ITexture* tex,
        SamplerState samp,
        const float* location,
        const float* clamp = nullptr) = 0;
    virtual void WriteSamplerFeedbackBias(
        ITexture* tex,
        SamplerState samp,
        const float* location,
        float bias,
        const float* clamp = nullptr) = 0;
    virtual void WriteSamplerFeedbackGrad(
        ITexture* tex,
        SamplerState samp,
        const float* location,
        const float* ddx,
        const float* ddy,
        const float* clamp = nullptr) = 0;

    virtual void WriteSamplerFeedbackLevel(
        ITexture* tex,
        SamplerState samp,
        const float* location,
        float lod) = 0;
};

template<typename T>
struct FeedbackTexture2D
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    template<typename S>
    void WriteSamplerFeedback(Texture2D<S> tex, SamplerState samp, float2 location, float clamp)
    {
        texture->WriteSamplerFeedback(tex.texture, samp, &location.x, &clamp);
    }

    template<typename S>
    void WriteSamplerFeedbackBias(
        Texture2D<S> tex,
        SamplerState samp,
        float2 location,
        float bias,
        float clamp)
    {
        texture->WriteSamplerFeedbackBias(tex.texture, samp, &location.x, bias, &clamp);
    }

    template<typename S>
    void WriteSamplerFeedbackGrad(
        Texture2D<S> tex,
        SamplerState samp,
        float2 location,
        float2 ddx,
        float2 ddy,
        float clamp)
    {
        texture->WriteSamplerFeedbackGrad(tex.texture, samp, &location.x, &ddx.x, &ddy.x, &clamp);
    }

    // Level

    template<typename S>
    void WriteSamplerFeedbackLevel(Texture2D<S> tex, SamplerState samp, float2 location, float lod)
    {
        texture->WriteSamplerFeedbackLevel(tex.texture, samp, &location.x, lod);
    }

    // Without Clamp
    template<typename S>
    void WriteSamplerFeedback(Texture2D<S> tex, SamplerState samp, float2 location)
    {
        texture->WriteSamplerFeedback(tex.texture, samp, &location.x);
    }

    template<typename S>
    void WriteSamplerFeedbackBias(Texture2D<S> tex, SamplerState samp, float2 location, float bias)
    {
        texture->WriteSamplerFeedbackBias(tex.texture, samp, &location.x, bias);
    }

    template<typename S>
    void WriteSamplerFeedbackGrad(
        Texture2D<S> tex,
        SamplerState samp,
        float2 location,
        float2 ddx,
        float2 ddy)
    {
        texture->WriteSamplerFeedbackGrad(tex.texture, samp, &location.x, &ddx.x, &ddy.x);
    }

    IFeedbackTexture* texture;
};

template<typename T>
struct FeedbackTexture2DArray
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight, uint32_t* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outElements,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight, float* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outElements,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    template<typename S>
    void WriteSamplerFeedback(
        Texture2DArray<S> texArray,
        SamplerState samp,
        float3 location,
        float clamp)
    {
        texture->WriteSamplerFeedback(texArray.texture, samp, &location.x, &clamp);
    }

    template<typename S>
    void WriteSamplerFeedbackBias(
        Texture2DArray<S> texArray,
        SamplerState samp,
        float3 location,
        float bias,
        float clamp)
    {
        texture->WriteSamplerFeedbackBias(texArray.texture, samp, &location.x, bias, &clamp);
    }

    template<typename S>
    void WriteSamplerFeedbackGrad(
        Texture2DArray<S> texArray,
        SamplerState samp,
        float3 location,
        float3 ddx,
        float3 ddy,
        float clamp)
    {
        texture
            ->WriteSamplerFeedbackGrad(texArray.texture, samp, &location.x, &ddx.x, &ddy.x, &clamp);
    }

    // Level
    template<typename S>
    void WriteSamplerFeedbackLevel(
        Texture2DArray<S> texArray,
        SamplerState samp,
        float3 location,
        float lod)
    {
        texture->WriteSamplerFeedbackLevel(texArray.texture, samp, &location.x, lod);
    }

    // Without Clamp

    template<typename S>
    void WriteSamplerFeedback(Texture2DArray<S> texArray, SamplerState samp, float3 location)
    {
        texture->WriteSamplerFeedback(texArray.texture, samp, &location.x);
    }

    template<typename S>
    void WriteSamplerFeedbackBias(
        Texture2DArray<S> texArray,
        SamplerState samp,
        float3 location,
        float bias)
    {
        texture->WriteSamplerFeedbackBias(texArray.texture, samp, &location.x, bias);
    }

    template<typename S>
    void WriteSamplerFeedbackGrad(
        Texture2DArray<S> texArray,
        SamplerState samp,
        float3 location,
        float3 ddx,
        float3 ddy)
    {
        texture->WriteSamplerFeedbackGrad(texArray.texture, samp, &location.x, &ddx.x, &ddy.x);
    }

    IFeedbackTexture* texture;
};

/* Varying input for Compute */

/* Used when running a single thread */
struct ComputeThreadVaryingInput
{
    uint3 groupID;
    uint3 groupThreadID;
};

struct ComputeVaryingInput
{
    uint3 startGroupID; ///< start groupID
    uint3 endGroupID;   ///< Non inclusive end groupID
};

// The uniformEntryPointParams and uniformState must be set to structures that match layout that the
// kernel expects. This can be determined via reflection for example.

typedef void (*ComputeThreadFunc)(
    ComputeThreadVaryingInput* varyingInput,
    void* uniformEntryPointParams,
    void* uniformState);
typedef void (*ComputeFunc)(
    ComputeVaryingInput* varyingInput,
    void* uniformEntryPointParams,
    void* uniformState);

#ifdef SLANG_PRELUDE_NAMESPACE
}
#endif

#endif


// Atomic helpers for the CPU target. Needed so kIROp_AtomicAdd and
// friends can lower to native atomic operations on the host - matching
// the semantics of HLSL `InterlockedAdd`, SPIR-V `OpAtomicIAdd`, etc.
// Uses compiler builtins so no <atomic> header is required.
//
// Contract for every `_slang_atomic_add_*` helper below (u32/i32/u64/
// i64): atomically add `val` to `*ptr` and return the PRIOR value (the
// value before the add), matching HLSL `InterlockedAdd` and GLSL
// `atomicAdd`. The 64-bit MSVC variants reinterpret the pointer as
// `volatile long long*` for `_InterlockedExchangeAdd64`, which is sound
// because `sizeof(long long) == 8` (asserted below).
//
// Compiler coverage:
//   - MSVC:            `_InterlockedExchangeAdd`
//   - GCC / Clang:     `__atomic_fetch_add`
//   - SNC / GHS etc.:  falls back to a non-atomic load/add/store.
//     These are console / embedded toolchains not known to ship
//     `__atomic_*` builtins; the fallback is racy under concurrency
//     but keeps the prelude buildable there. Platforms that need
//     real atomics on those compilers can add a bespoke branch.
// === MSVC implementations ===
//
// Use the `_Interlocked*` intrinsics. The 32-bit overload operates on
// `long`; the 64-bit overload operates on `long long`. The static
// asserts above pin the expected widths.
#if SLANG_VC
#include <intrin.h>
static_assert(
    sizeof(long) == 4,
    "_InterlockedExchangeAdd uses `long`; MSVC LLP64 requires sizeof(long)==4");
static_assert(
    sizeof(long long) == 8,
    "_InterlockedExchangeAdd64 uses `long long`; expected 8-byte width");
static inline uint32_t _slang_atomic_add_u32(uint32_t* ptr, uint32_t val)
{
    // Returns the PRIOR value, matching HLSL InterlockedAdd and GLSL atomicAdd.
    return static_cast<uint32_t>(
        _InterlockedExchangeAdd(reinterpret_cast<volatile long*>(ptr), static_cast<long>(val)));
}
static inline int32_t _slang_atomic_add_i32(int32_t* ptr, int32_t val)
{
    return static_cast<int32_t>(
        _InterlockedExchangeAdd(reinterpret_cast<volatile long*>(ptr), static_cast<long>(val)));
}
static inline uint64_t _slang_atomic_add_u64(uint64_t* ptr, uint64_t val)
{
    return static_cast<uint64_t>(_InterlockedExchangeAdd64(
        reinterpret_cast<volatile long long*>(ptr),
        static_cast<long long>(val)));
}
static inline int64_t _slang_atomic_add_i64(int64_t* ptr, int64_t val)
{
    return static_cast<int64_t>(_InterlockedExchangeAdd64(
        reinterpret_cast<volatile long long*>(ptr),
        static_cast<long long>(val)));
}
// === GCC / Clang implementations ===
//
// Use the `__atomic_fetch_add` built-in with relaxed ordering; the
// IR-side AtomicAdd emit ignores the memory-order operand and lets
// each toolchain pick its native default.
#elif SLANG_GCC || SLANG_CLANG
static inline uint32_t _slang_atomic_add_u32(uint32_t* ptr, uint32_t val)
{
    return __atomic_fetch_add(ptr, val, __ATOMIC_RELAXED);
}
static inline int32_t _slang_atomic_add_i32(int32_t* ptr, int32_t val)
{
    return __atomic_fetch_add(ptr, val, __ATOMIC_RELAXED);
}
static inline uint64_t _slang_atomic_add_u64(uint64_t* ptr, uint64_t val)
{
    return __atomic_fetch_add(ptr, val, __ATOMIC_RELAXED);
}
static inline int64_t _slang_atomic_add_i64(int64_t* ptr, int64_t val)
{
    return __atomic_fetch_add(ptr, val, __ATOMIC_RELAXED);
}
// === Non-atomic fallback implementations ===
//
// For compilers without a known atomic builtin (Sony SNC, Green Hills
// MULTI, etc.). Racy under concurrent invocation but keeps the
// prelude compilable; CPU-target coverage on these platforms is
// single-threaded in practice.
#else
static inline uint32_t _slang_atomic_add_u32(uint32_t* ptr, uint32_t val)
{
    uint32_t old = *ptr;
    *ptr = old + val;
    return old;
}
static inline int32_t _slang_atomic_add_i32(int32_t* ptr, int32_t val)
{
    int32_t old = *ptr;
    *ptr = old + val;
    return old;
}
static inline uint64_t _slang_atomic_add_u64(uint64_t* ptr, uint64_t val)
{
    uint64_t old = *ptr;
    *ptr = old + val;
    return old;
}
static inline int64_t _slang_atomic_add_i64(int64_t* ptr, int64_t val)
{
    int64_t old = *ptr;
    *ptr = old + val;
    return old;
}
#endif

// TODO(JS): Hack! Output C++ code from slang can copy uninitialized variables.
#if defined(_MSC_VER)
#pragma warning(disable : 4700)
#endif

#ifndef SLANG_UNROLL
#define SLANG_UNROLL
#endif

#endif

#ifdef SLANG_PRELUDE_NAMESPACE
using namespace SLANG_PRELUDE_NAMESPACE;
#endif


#line 1 "slang/similarity_hessian_block.slang"
struct df_0
{
    float hi_0;
    float lo_0;
};


#line 6
struct SimHessParams_0
{
    uint32_t count_0;
    float sign_0;
};


#line 6
struct GlobalParams_0
{
    StructuredBuffer<df_0> pos_0;
    StructuredBuffer<uint32_t> hinge_v_0;
    StructuredBuffer<df_0> coef_0;
    RWStructuredBuffer<df_0> blocks_0;
    SimHessParams_0* params_0;
};


#line 6
struct KernelContext_0
{
    GlobalParams_0* globalParams_0;
};


#line 22
static df_0 df_make_0(float h_0, float l_0)
{

#line 23
    df_0 r_0;
    (&r_0)->hi_0 = h_0;
    (&r_0)->lo_0 = l_0;
    return r_0;
}


#line 63
static df_0 df_neg_0(df_0 * x_0)
{

#line 64
    return df_make_0(- x_0->hi_0, - x_0->lo_0);
}


#line 33
static df_0 two_sum_0(float a_0, float b_0)
{

#line 34
    precise float s_0 = a_0 + b_0;
    precise float bb_0 = s_0 - a_0;
    precise float ea_0 = a_0 - (s_0 - bb_0);
    precise float eb_0 = b_0 - bb_0;
    precise float e_0 = ea_0 + eb_0;
    return df_make_0(s_0, e_0);
}

static df_0 quick_two_sum_0(float a_1, float b_1)
{

#line 43
    precise float s_1 = a_1 + b_1;
    precise float e_1 = b_1 - (s_1 - a_1);
    return df_make_0(s_1, e_1);
}


#line 54
static df_0 df_add_0(df_0 * x_1, df_0 * y_0)
{

#line 55
    df_0 s_2 = two_sum_0(x_1->hi_0, y_0->hi_0);
    df_0 t_0 = two_sum_0(x_1->lo_0, y_0->lo_0);
    precise float sl_0 = s_2.lo_0 + t_0.hi_0;
    df_0 u_0 = quick_two_sum_0(s_2.hi_0, sl_0);
    precise float ul_0 = u_0.lo_0 + t_0.lo_0;
    return quick_two_sum_0(u_0.hi_0, ul_0);
}


#line 48
static df_0 two_prod_0(float a_2, float b_2)
{

#line 49
    precise float p_0 = a_2 * b_2;
    precise float e_2 = (F32_fma((a_2), (b_2), (- p_0)));
    return df_make_0(p_0, e_2);
}


#line 71
static df_0 df_mul_0(df_0 * x_2, df_0 * y_1)
{

#line 72
    df_0 p_1 = two_prod_0(x_2->hi_0, y_1->hi_0);
    precise float c_0 = x_2->hi_0 * y_1->lo_0;
    precise float d_0 = x_2->lo_0 * y_1->hi_0;
    precise float cd_0 = c_0 + d_0;
    precise float pl_0 = p_1.lo_0 + cd_0;
    return quick_two_sum_0(p_1.hi_0, pl_0);
}

static df_0 df_mulf_0(df_0 * x_3, float c_1)
{

#line 81
    df_0 p_2 = two_prod_0(x_3->hi_0, c_1);
    precise float d_1 = x_3->lo_0 * c_1;
    precise float pl_1 = p_2.lo_0 + d_1;
    return quick_two_sum_0(p_2.hi_0, pl_1);
}


#line 125
void _main_0(void* _S1, void* entryPointParams_0, void* globalParams_1)
{

#line 125
    ComputeThreadVaryingInput * _S2 = (slang_bit_cast<ComputeThreadVaryingInput *>(_S1));

#line 125
    KernelContext_0 kernelContext_0;

#line 125
    (&kernelContext_0)->globalParams_0 = (slang_bit_cast<GlobalParams_0*>(globalParams_1));
    uint32_t k_0 = (_S2->groupID * Vector<uint32_t, 3> (64U, 1U, 1U) + _S2->groupThreadID).x;
    if(k_0 >= ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->count_0))
    {

#line 128
        return;
    }
    uint32_t hb_0 = 4U * k_0;
    uint32_t cb_0 = 6U * k_0;
    uint32_t ob_0 = 144U * k_0;

    df_0 p1_0 = (&kernelContext_0)->globalParams_0->pos_0.Load(3U * (&kernelContext_0)->globalParams_0->hinge_v_0.Load(hb_0) + 1U);
    df_0 p2_0 = (&kernelContext_0)->globalParams_0->pos_0.Load(3U * (&kernelContext_0)->globalParams_0->hinge_v_0.Load(hb_0) + 2U);
    uint32_t _S3 = hb_0 + 1U;

#line 136
    df_0 p3_0 = (&kernelContext_0)->globalParams_0->pos_0.Load(3U * (&kernelContext_0)->globalParams_0->hinge_v_0.Load(_S3));
    df_0 p4_0 = (&kernelContext_0)->globalParams_0->pos_0.Load(3U * (&kernelContext_0)->globalParams_0->hinge_v_0.Load(_S3) + 1U);
    df_0 p5_0 = (&kernelContext_0)->globalParams_0->pos_0.Load(3U * (&kernelContext_0)->globalParams_0->hinge_v_0.Load(_S3) + 2U);
    uint32_t _S4 = hb_0 + 2U;

#line 139
    df_0 p6_0 = (&kernelContext_0)->globalParams_0->pos_0.Load(3U * (&kernelContext_0)->globalParams_0->hinge_v_0.Load(_S4));
    df_0 p7_0 = (&kernelContext_0)->globalParams_0->pos_0.Load(3U * (&kernelContext_0)->globalParams_0->hinge_v_0.Load(_S4) + 1U);
    df_0 p8_0 = (&kernelContext_0)->globalParams_0->pos_0.Load(3U * (&kernelContext_0)->globalParams_0->hinge_v_0.Load(_S4) + 2U);
    uint32_t _S5 = hb_0 + 3U;

#line 142
    df_0 p9_0 = (&kernelContext_0)->globalParams_0->pos_0.Load(3U * (&kernelContext_0)->globalParams_0->hinge_v_0.Load(_S5));
    df_0 p10_0 = (&kernelContext_0)->globalParams_0->pos_0.Load(3U * (&kernelContext_0)->globalParams_0->hinge_v_0.Load(_S5) + 1U);
    df_0 p11_0 = (&kernelContext_0)->globalParams_0->pos_0.Load(3U * (&kernelContext_0)->globalParams_0->hinge_v_0.Load(_S5) + 2U);
    df_0 c0_0 = (&kernelContext_0)->globalParams_0->coef_0.Load(cb_0);
    df_0 c1_0 = (&kernelContext_0)->globalParams_0->coef_0.Load(cb_0 + 1U);
    df_0 c2_0 = (&kernelContext_0)->globalParams_0->coef_0.Load(cb_0 + 2U);
    df_0 c3_0 = (&kernelContext_0)->globalParams_0->coef_0.Load(cb_0 + 3U);
    df_0 c4_0 = (&kernelContext_0)->globalParams_0->coef_0.Load(cb_0 + 4U);
    df_0 c5_0 = (&kernelContext_0)->globalParams_0->coef_0.Load(cb_0 + 5U);

#line 150
    df_0 _S6 = (&kernelContext_0)->globalParams_0->pos_0.Load(3U * (&kernelContext_0)->globalParams_0->hinge_v_0.Load(hb_0));

#line 150
    df_0 _S7 = df_neg_0(&_S6);

#line 150
    df_0 _S8 = p3_0;

#line 150
    df_0 _S9 = _S7;

#line 150
    df_0 _S10 = df_add_0(&_S8, &_S9);

#line 150
    df_0 _S11 = p1_0;

#line 150
    df_0 _S12 = df_neg_0(&_S11);

#line 150
    df_0 _S13 = p4_0;

#line 150
    df_0 _S14 = _S12;

#line 150
    df_0 _S15 = df_add_0(&_S13, &_S14);

#line 150
    df_0 _S16 = p2_0;

#line 150
    df_0 _S17 = df_neg_0(&_S16);

#line 150
    df_0 _S18 = p5_0;

#line 150
    df_0 _S19 = _S17;

#line 150
    df_0 _S20 = df_add_0(&_S18, &_S19);

#line 150
    df_0 _S21 = p6_0;

#line 150
    df_0 _S22 = _S7;

#line 150
    df_0 _S23 = df_add_0(&_S21, &_S22);

#line 150
    df_0 _S24 = p7_0;

#line 150
    df_0 _S25 = _S12;

#line 150
    df_0 _S26 = df_add_0(&_S24, &_S25);

#line 150
    df_0 _S27 = p8_0;

#line 150
    df_0 _S28 = _S17;

#line 150
    df_0 _S29 = df_add_0(&_S27, &_S28);

#line 150
    df_0 _S30 = p9_0;

#line 150
    df_0 _S31 = _S7;

#line 150
    df_0 _S32 = df_add_0(&_S30, &_S31);

#line 150
    df_0 _S33 = p10_0;

#line 150
    df_0 _S34 = _S12;

#line 150
    df_0 _S35 = df_add_0(&_S33, &_S34);

#line 150
    df_0 _S36 = p11_0;

#line 150
    df_0 _S37 = _S17;

#line 150
    df_0 _S38 = df_add_0(&_S36, &_S37);

#line 150
    df_0 _S39 = _S15;

#line 150
    df_0 _S40 = _S29;

#line 150
    df_0 _S41 = df_mul_0(&_S39, &_S40);

#line 150
    df_0 _S42 = _S29;

#line 150
    df_0 _S43 = df_neg_0(&_S42);

#line 150
    df_0 _S44 = _S15;

#line 150
    df_0 _S45 = df_neg_0(&_S44);

#line 150
    df_0 _S46 = _S20;

#line 150
    df_0 _S47 = _S26;

#line 150
    df_0 _S48 = df_mul_0(&_S46, &_S47);

#line 150
    df_0 _S49 = _S20;

#line 150
    df_0 _S50 = df_neg_0(&_S49);

#line 150
    df_0 _S51 = _S26;

#line 150
    df_0 _S52 = df_neg_0(&_S51);

#line 150
    df_0 _S53 = _S48;

#line 150
    df_0 _S54 = df_neg_0(&_S53);

#line 150
    df_0 _S55 = _S50;

#line 150
    df_0 _S56 = df_neg_0(&_S55);

#line 150
    df_0 _S57 = _S52;

#line 150
    df_0 _S58 = df_neg_0(&_S57);

#line 150
    df_0 _S59 = _S41;

#line 150
    df_0 _S60 = _S54;

#line 150
    df_0 _S61 = df_add_0(&_S59, &_S60);

#line 150
    df_0 _S62 = _S43;

#line 150
    df_0 _S63 = _S56;

#line 150
    df_0 _S64 = df_add_0(&_S62, &_S63);

#line 150
    df_0 _S65 = _S45;

#line 150
    df_0 _S66 = _S58;

#line 150
    df_0 _S67 = df_add_0(&_S65, &_S66);

#line 183
    df_0 _S68 = df_make_0(-1.0f, 0.0f);

#line 183
    df_0 _S69 = df_make_0(1.0f, 0.0f);

#line 183
    df_0 _S70 = _S68;

#line 183
    df_0 _S71 = df_add_0(&_S69, &_S70);

#line 183
    df_0 _S72 = _S20;

#line 183
    df_0 _S73 = _S23;

#line 183
    df_0 _S74 = df_mul_0(&_S72, &_S73);

#line 183
    df_0 _S75 = _S23;

#line 183
    df_0 _S76 = df_neg_0(&_S75);

#line 183
    df_0 _S77 = _S10;

#line 183
    df_0 _S78 = _S29;

#line 183
    df_0 _S79 = df_mul_0(&_S77, &_S78);

#line 183
    df_0 _S80 = _S10;

#line 183
    df_0 _S81 = df_neg_0(&_S80);

#line 183
    df_0 _S82 = _S79;

#line 183
    df_0 _S83 = df_neg_0(&_S82);

#line 183
    df_0 _S84 = _S43;

#line 183
    df_0 _S85 = df_neg_0(&_S84);

#line 183
    df_0 _S86 = _S81;

#line 183
    df_0 _S87 = df_neg_0(&_S86);

#line 183
    df_0 _S88 = _S74;

#line 183
    df_0 _S89 = _S83;

#line 183
    df_0 _S90 = df_add_0(&_S88, &_S89);

#line 183
    df_0 _S91 = _S50;

#line 183
    df_0 _S92 = _S85;

#line 183
    df_0 _S93 = df_add_0(&_S91, &_S92);

#line 183
    df_0 _S94 = _S76;

#line 183
    df_0 _S95 = _S87;

#line 183
    df_0 _S96 = df_add_0(&_S94, &_S95);

#line 183
    df_0 _S97 = _S10;

#line 183
    df_0 _S98 = _S26;

#line 183
    df_0 _S99 = df_mul_0(&_S97, &_S98);

#line 183
    df_0 _S100 = _S15;

#line 183
    df_0 _S101 = _S23;

#line 183
    df_0 _S102 = df_mul_0(&_S100, &_S101);

#line 183
    df_0 _S103 = _S102;

#line 183
    df_0 _S104 = df_neg_0(&_S103);

#line 183
    df_0 _S105 = _S45;

#line 183
    df_0 _S106 = df_neg_0(&_S105);

#line 183
    df_0 _S107 = _S76;

#line 183
    df_0 _S108 = df_neg_0(&_S107);

#line 183
    df_0 _S109 = _S99;

#line 183
    df_0 _S110 = _S104;

#line 183
    df_0 _S111 = df_add_0(&_S109, &_S110);

#line 183
    df_0 _S112 = _S52;

#line 183
    df_0 _S113 = _S106;

#line 183
    df_0 _S114 = df_add_0(&_S112, &_S113);

#line 183
    df_0 _S115 = _S81;

#line 183
    df_0 _S116 = _S108;

#line 183
    df_0 _S117 = df_add_0(&_S115, &_S116);

#line 183
    df_0 _S118 = _S35;

#line 183
    df_0 _S119 = _S20;

#line 183
    df_0 _S120 = df_mul_0(&_S118, &_S119);

#line 183
    df_0 _S121 = _S35;

#line 183
    df_0 _S122 = df_neg_0(&_S121);

#line 183
    df_0 _S123 = _S38;

#line 183
    df_0 _S124 = _S15;

#line 183
    df_0 _S125 = df_mul_0(&_S123, &_S124);

#line 183
    df_0 _S126 = _S38;

#line 183
    df_0 _S127 = df_neg_0(&_S126);

#line 183
    df_0 _S128 = _S125;

#line 183
    df_0 _S129 = df_neg_0(&_S128);

#line 183
    df_0 _S130 = _S127;

#line 183
    df_0 _S131 = df_neg_0(&_S130);

#line 183
    df_0 _S132 = _S120;

#line 183
    df_0 _S133 = _S129;

#line 183
    df_0 _S134 = df_add_0(&_S132, &_S133);

#line 183
    df_0 _S135 = _S50;

#line 183
    df_0 _S136 = _S131;

#line 183
    df_0 _S137 = df_add_0(&_S135, &_S136);

#line 183
    df_0 _S138 = _S122;

#line 183
    df_0 _S139 = _S106;

#line 183
    df_0 _S140 = df_add_0(&_S138, &_S139);

#line 183
    df_0 _S141 = _S38;

#line 183
    df_0 _S142 = _S10;

#line 183
    df_0 _S143 = df_mul_0(&_S141, &_S142);

#line 183
    df_0 _S144 = _S32;

#line 183
    df_0 _S145 = _S20;

#line 183
    df_0 _S146 = df_mul_0(&_S144, &_S145);

#line 183
    df_0 _S147 = _S32;

#line 183
    df_0 _S148 = df_neg_0(&_S147);

#line 183
    df_0 _S149 = _S146;

#line 183
    df_0 _S150 = df_neg_0(&_S149);

#line 183
    df_0 _S151 = _S148;

#line 183
    df_0 _S152 = df_neg_0(&_S151);

#line 183
    df_0 _S153 = _S143;

#line 183
    df_0 _S154 = _S150;

#line 183
    df_0 _S155 = df_add_0(&_S153, &_S154);

#line 183
    df_0 _S156 = _S127;

#line 183
    df_0 _S157 = _S56;

#line 183
    df_0 _S158 = df_add_0(&_S156, &_S157);

#line 183
    df_0 _S159 = _S81;

#line 183
    df_0 _S160 = _S152;

#line 183
    df_0 _S161 = df_add_0(&_S159, &_S160);

#line 183
    df_0 _S162 = _S32;

#line 183
    df_0 _S163 = _S15;

#line 183
    df_0 _S164 = df_mul_0(&_S162, &_S163);

#line 183
    df_0 _S165 = _S35;

#line 183
    df_0 _S166 = _S10;

#line 183
    df_0 _S167 = df_mul_0(&_S165, &_S166);

#line 183
    df_0 _S168 = _S167;

#line 183
    df_0 _S169 = df_neg_0(&_S168);

#line 183
    df_0 _S170 = _S122;

#line 183
    df_0 _S171 = df_neg_0(&_S170);

#line 183
    df_0 _S172 = _S164;

#line 183
    df_0 _S173 = _S169;

#line 183
    df_0 _S174 = df_add_0(&_S172, &_S173);

#line 183
    df_0 _S175 = _S45;

#line 183
    df_0 _S176 = _S171;

#line 183
    df_0 _S177 = df_add_0(&_S175, &_S176);

#line 183
    df_0 _S178 = _S148;

#line 183
    df_0 _S179 = _S87;

#line 183
    df_0 _S180 = df_add_0(&_S178, &_S179);

#line 183
    df_0 _S181 = c0_0;

#line 183
    df_0 _S182 = _S10;

#line 183
    df_0 _S183 = df_mul_0(&_S181, &_S182);

#line 183
    df_0 _S184 = c0_0;

#line 183
    df_0 _S185 = df_neg_0(&_S184);

#line 183
    df_0 _S186 = c0_0;

#line 183
    df_0 _S187 = _S15;

#line 183
    df_0 _S188 = df_mul_0(&_S186, &_S187);

#line 183
    df_0 _S189 = c0_0;

#line 183
    df_0 _S190 = _S20;

#line 183
    df_0 _S191 = df_mul_0(&_S189, &_S190);

#line 183
    df_0 _S192 = c1_0;

#line 183
    df_0 _S193 = _S23;

#line 183
    df_0 _S194 = df_mul_0(&_S192, &_S193);

#line 183
    df_0 _S195 = c1_0;

#line 183
    df_0 _S196 = df_neg_0(&_S195);

#line 183
    df_0 _S197 = c1_0;

#line 183
    df_0 _S198 = _S26;

#line 183
    df_0 _S199 = df_mul_0(&_S197, &_S198);

#line 183
    df_0 _S200 = c1_0;

#line 183
    df_0 _S201 = _S29;

#line 183
    df_0 _S202 = df_mul_0(&_S200, &_S201);

#line 183
    df_0 _S203 = _S183;

#line 183
    df_0 _S204 = _S194;

#line 183
    df_0 _S205 = df_add_0(&_S203, &_S204);

#line 183
    df_0 _S206 = _S185;

#line 183
    df_0 _S207 = _S196;

#line 183
    df_0 _S208 = df_add_0(&_S206, &_S207);

#line 183
    df_0 _S209 = _S188;

#line 183
    df_0 _S210 = _S199;

#line 183
    df_0 _S211 = df_add_0(&_S209, &_S210);

#line 183
    df_0 _S212 = _S191;

#line 183
    df_0 _S213 = _S202;

#line 183
    df_0 _S214 = df_add_0(&_S212, &_S213);

#line 183
    df_0 _S215 = c2_0;

#line 183
    df_0 _S216 = _S61;

#line 183
    df_0 _S217 = df_mul_0(&_S215, &_S216);

#line 183
    df_0 _S218 = c2_0;

#line 183
    df_0 _S219 = _S64;

#line 183
    df_0 _S220 = df_mul_0(&_S218, &_S219);

#line 183
    df_0 _S221 = c2_0;

#line 183
    df_0 _S222 = _S67;

#line 183
    df_0 _S223 = df_mul_0(&_S221, &_S222);

#line 183
    df_0 _S224 = c2_0;

#line 183
    df_0 _S225 = _S29;

#line 183
    df_0 _S226 = df_mul_0(&_S224, &_S225);

#line 183
    df_0 _S227 = c2_0;

#line 183
    df_0 _S228 = _S52;

#line 183
    df_0 _S229 = df_mul_0(&_S227, &_S228);

#line 183
    df_0 _S230 = c2_0;

#line 183
    df_0 _S231 = _S50;

#line 183
    df_0 _S232 = df_mul_0(&_S230, &_S231);

#line 183
    df_0 _S233 = c2_0;

#line 183
    df_0 _S234 = _S15;

#line 183
    df_0 _S235 = df_mul_0(&_S233, &_S234);

#line 183
    df_0 _S236 = c2_0;

#line 183
    df_0 _S237 = _S71;

#line 183
    df_0 _S238 = df_mul_0(&_S236, &_S237);

#line 183
    df_0 _S239 = c2_0;

#line 183
    df_0 _S240 = df_neg_0(&_S239);

#line 183
    df_0 _S241 = c2_0;

#line 183
    df_0 _S242 = _S90;

#line 183
    df_0 _S243 = df_mul_0(&_S241, &_S242);

#line 183
    df_0 _S244 = c2_0;

#line 183
    df_0 _S245 = _S93;

#line 183
    df_0 _S246 = df_mul_0(&_S244, &_S245);

#line 183
    df_0 _S247 = c2_0;

#line 183
    df_0 _S248 = _S96;

#line 183
    df_0 _S249 = df_mul_0(&_S247, &_S248);

#line 183
    df_0 _S250 = c2_0;

#line 183
    df_0 _S251 = _S43;

#line 183
    df_0 _S252 = df_mul_0(&_S250, &_S251);

#line 183
    df_0 _S253 = c2_0;

#line 183
    df_0 _S254 = _S23;

#line 183
    df_0 _S255 = df_mul_0(&_S253, &_S254);

#line 183
    df_0 _S256 = c2_0;

#line 183
    df_0 _S257 = _S20;

#line 183
    df_0 _S258 = df_mul_0(&_S256, &_S257);

#line 183
    df_0 _S259 = c2_0;

#line 183
    df_0 _S260 = _S81;

#line 183
    df_0 _S261 = df_mul_0(&_S259, &_S260);

#line 183
    df_0 _S262 = c2_0;

#line 183
    df_0 _S263 = _S111;

#line 183
    df_0 _S264 = df_mul_0(&_S262, &_S263);

#line 183
    df_0 _S265 = c2_0;

#line 183
    df_0 _S266 = _S114;

#line 183
    df_0 _S267 = df_mul_0(&_S265, &_S266);

#line 183
    df_0 _S268 = c2_0;

#line 183
    df_0 _S269 = _S117;

#line 183
    df_0 _S270 = df_mul_0(&_S268, &_S269);

#line 183
    df_0 _S271 = c2_0;

#line 183
    df_0 _S272 = _S26;

#line 183
    df_0 _S273 = df_mul_0(&_S271, &_S272);

#line 183
    df_0 _S274 = c2_0;

#line 183
    df_0 _S275 = _S76;

#line 183
    df_0 _S276 = df_mul_0(&_S274, &_S275);

#line 183
    df_0 _S277 = c2_0;

#line 183
    df_0 _S278 = _S45;

#line 183
    df_0 _S279 = df_mul_0(&_S277, &_S278);

#line 183
    df_0 _S280 = c2_0;

#line 183
    df_0 _S281 = _S10;

#line 183
    df_0 _S282 = df_mul_0(&_S280, &_S281);

#line 183
    df_0 _S283 = _S205;

#line 183
    df_0 _S284 = _S217;

#line 183
    df_0 _S285 = df_add_0(&_S283, &_S284);

#line 183
    df_0 _S286 = _S211;

#line 183
    df_0 _S287 = _S243;

#line 183
    df_0 _S288 = df_add_0(&_S286, &_S287);

#line 183
    df_0 _S289 = _S214;

#line 183
    df_0 _S290 = _S264;

#line 183
    df_0 _S291 = df_add_0(&_S289, &_S290);

#line 183
    df_0 _S292 = c3_0;

#line 183
    df_0 _S293 = _S32;

#line 183
    df_0 _S294 = df_mul_0(&_S292, &_S293);

#line 183
    df_0 _S295 = c3_0;

#line 183
    df_0 _S296 = df_neg_0(&_S295);

#line 183
    df_0 _S297 = c3_0;

#line 183
    df_0 _S298 = _S35;

#line 183
    df_0 _S299 = df_mul_0(&_S297, &_S298);

#line 183
    df_0 _S300 = c3_0;

#line 183
    df_0 _S301 = _S38;

#line 183
    df_0 _S302 = df_mul_0(&_S300, &_S301);

#line 183
    df_0 _S303 = c4_0;

#line 183
    df_0 _S304 = _S10;

#line 183
    df_0 _S305 = df_mul_0(&_S303, &_S304);

#line 183
    df_0 _S306 = c4_0;

#line 183
    df_0 _S307 = df_neg_0(&_S306);

#line 183
    df_0 _S308 = c4_0;

#line 183
    df_0 _S309 = _S15;

#line 183
    df_0 _S310 = df_mul_0(&_S308, &_S309);

#line 183
    df_0 _S311 = c4_0;

#line 183
    df_0 _S312 = _S20;

#line 183
    df_0 _S313 = df_mul_0(&_S311, &_S312);

#line 183
    df_0 _S314 = _S294;

#line 183
    df_0 _S315 = _S305;

#line 183
    df_0 _S316 = df_add_0(&_S314, &_S315);

#line 183
    df_0 _S317 = _S296;

#line 183
    df_0 _S318 = _S307;

#line 183
    df_0 _S319 = df_add_0(&_S317, &_S318);

#line 183
    df_0 _S320 = _S299;

#line 183
    df_0 _S321 = _S310;

#line 183
    df_0 _S322 = df_add_0(&_S320, &_S321);

#line 183
    df_0 _S323 = _S302;

#line 183
    df_0 _S324 = _S313;

#line 183
    df_0 _S325 = df_add_0(&_S323, &_S324);

#line 183
    df_0 _S326 = c5_0;

#line 183
    df_0 _S327 = _S134;

#line 183
    df_0 _S328 = df_mul_0(&_S326, &_S327);

#line 183
    df_0 _S329 = c5_0;

#line 183
    df_0 _S330 = _S137;

#line 183
    df_0 _S331 = df_mul_0(&_S329, &_S330);

#line 183
    df_0 _S332 = c5_0;

#line 183
    df_0 _S333 = _S140;

#line 183
    df_0 _S334 = df_mul_0(&_S332, &_S333);

#line 183
    df_0 _S335 = c5_0;

#line 183
    df_0 _S336 = _S127;

#line 183
    df_0 _S337 = df_mul_0(&_S335, &_S336);

#line 183
    df_0 _S338 = c5_0;

#line 183
    df_0 _S339 = _S35;

#line 183
    df_0 _S340 = df_mul_0(&_S338, &_S339);

#line 183
    df_0 _S341 = c5_0;

#line 183
    df_0 _S342 = _S20;

#line 183
    df_0 _S343 = df_mul_0(&_S341, &_S342);

#line 183
    df_0 _S344 = c5_0;

#line 183
    df_0 _S345 = _S45;

#line 183
    df_0 _S346 = df_mul_0(&_S344, &_S345);

#line 183
    df_0 _S347 = c5_0;

#line 183
    df_0 _S348 = _S71;

#line 183
    df_0 _S349 = df_mul_0(&_S347, &_S348);

#line 183
    df_0 _S350 = c5_0;

#line 183
    df_0 _S351 = df_neg_0(&_S350);

#line 183
    df_0 _S352 = c5_0;

#line 183
    df_0 _S353 = _S155;

#line 183
    df_0 _S354 = df_mul_0(&_S352, &_S353);

#line 183
    df_0 _S355 = c5_0;

#line 183
    df_0 _S356 = _S158;

#line 183
    df_0 _S357 = df_mul_0(&_S355, &_S356);

#line 183
    df_0 _S358 = c5_0;

#line 183
    df_0 _S359 = _S161;

#line 183
    df_0 _S360 = df_mul_0(&_S358, &_S359);

#line 183
    df_0 _S361 = c5_0;

#line 183
    df_0 _S362 = _S38;

#line 183
    df_0 _S363 = df_mul_0(&_S361, &_S362);

#line 183
    df_0 _S364 = c5_0;

#line 183
    df_0 _S365 = _S148;

#line 183
    df_0 _S366 = df_mul_0(&_S364, &_S365);

#line 183
    df_0 _S367 = c5_0;

#line 183
    df_0 _S368 = _S50;

#line 183
    df_0 _S369 = df_mul_0(&_S367, &_S368);

#line 183
    df_0 _S370 = c5_0;

#line 183
    df_0 _S371 = _S10;

#line 183
    df_0 _S372 = df_mul_0(&_S370, &_S371);

#line 183
    df_0 _S373 = c5_0;

#line 183
    df_0 _S374 = _S174;

#line 183
    df_0 _S375 = df_mul_0(&_S373, &_S374);

#line 183
    df_0 _S376 = c5_0;

#line 183
    df_0 _S377 = _S177;

#line 183
    df_0 _S378 = df_mul_0(&_S376, &_S377);

#line 183
    df_0 _S379 = c5_0;

#line 183
    df_0 _S380 = _S180;

#line 183
    df_0 _S381 = df_mul_0(&_S379, &_S380);

#line 183
    df_0 _S382 = c5_0;

#line 183
    df_0 _S383 = _S122;

#line 183
    df_0 _S384 = df_mul_0(&_S382, &_S383);

#line 183
    df_0 _S385 = c5_0;

#line 183
    df_0 _S386 = _S32;

#line 183
    df_0 _S387 = df_mul_0(&_S385, &_S386);

#line 183
    df_0 _S388 = c5_0;

#line 183
    df_0 _S389 = _S15;

#line 183
    df_0 _S390 = df_mul_0(&_S388, &_S389);

#line 183
    df_0 _S391 = c5_0;

#line 183
    df_0 _S392 = _S81;

#line 183
    df_0 _S393 = df_mul_0(&_S391, &_S392);

#line 183
    df_0 _S394 = _S316;

#line 183
    df_0 _S395 = _S328;

#line 183
    df_0 _S396 = df_add_0(&_S394, &_S395);

#line 183
    df_0 _S397 = _S322;

#line 183
    df_0 _S398 = _S354;

#line 183
    df_0 _S399 = df_add_0(&_S397, &_S398);

#line 183
    df_0 _S400 = _S325;

#line 183
    df_0 _S401 = _S375;

#line 183
    df_0 _S402 = df_add_0(&_S400, &_S401);

#line 183
    df_0 _S403 = _S396;

#line 183
    df_0 _S404 = df_neg_0(&_S403);

#line 183
    df_0 _S405 = _S319;

#line 183
    df_0 _S406 = df_neg_0(&_S405);

#line 183
    df_0 _S407 = _S331;

#line 183
    df_0 _S408 = df_neg_0(&_S407);

#line 183
    df_0 _S409 = _S334;

#line 183
    df_0 _S410 = df_neg_0(&_S409);

#line 183
    df_0 _S411 = _S337;

#line 183
    df_0 _S412 = df_neg_0(&_S411);

#line 183
    df_0 _S413 = _S340;

#line 183
    df_0 _S414 = df_neg_0(&_S413);

#line 183
    df_0 _S415 = _S343;

#line 183
    df_0 _S416 = df_neg_0(&_S415);

#line 183
    df_0 _S417 = _S346;

#line 183
    df_0 _S418 = df_neg_0(&_S417);

#line 183
    df_0 _S419 = _S349;

#line 183
    df_0 _S420 = df_neg_0(&_S419);

#line 183
    df_0 _S421 = _S351;

#line 183
    df_0 _S422 = df_neg_0(&_S421);

#line 183
    df_0 _S423 = _S285;

#line 183
    df_0 _S424 = _S404;

#line 183
    df_0 _S425 = df_add_0(&_S423, &_S424);

#line 183
    df_0 _S426 = _S208;

#line 183
    df_0 _S427 = _S406;

#line 183
    df_0 _S428 = df_add_0(&_S426, &_S427);

#line 183
    df_0 _S429 = _S220;

#line 183
    df_0 _S430 = _S408;

#line 183
    df_0 _S431 = df_add_0(&_S429, &_S430);

#line 183
    df_0 _S432 = _S223;

#line 183
    df_0 _S433 = _S410;

#line 183
    df_0 _S434 = df_add_0(&_S432, &_S433);

#line 183
    df_0 _S435 = c0_0;

#line 183
    df_0 _S436 = _S307;

#line 183
    df_0 _S437 = df_add_0(&_S435, &_S436);

#line 183
    df_0 _S438 = _S226;

#line 183
    df_0 _S439 = _S412;

#line 183
    df_0 _S440 = df_add_0(&_S438, &_S439);

#line 183
    df_0 _S441 = _S229;

#line 183
    df_0 _S442 = _S414;

#line 183
    df_0 _S443 = df_add_0(&_S441, &_S442);

#line 183
    df_0 _S444 = _S238;

#line 183
    df_0 _S445 = _S420;

#line 183
    df_0 _S446 = df_add_0(&_S444, &_S445);

#line 183
    df_0 _S447 = c2_0;

#line 183
    df_0 _S448 = _S422;

#line 183
    df_0 _S449 = df_add_0(&_S447, &_S448);

#line 183
    df_0 _S450 = _S240;

#line 183
    df_0 _S451 = _S351;

#line 183
    df_0 _S452 = df_add_0(&_S450, &_S451);

#line 183
    df_0 _S453 = _S399;

#line 183
    df_0 _S454 = df_neg_0(&_S453);

#line 183
    df_0 _S455 = _S357;

#line 183
    df_0 _S456 = df_neg_0(&_S455);

#line 183
    df_0 _S457 = _S360;

#line 183
    df_0 _S458 = df_neg_0(&_S457);

#line 183
    df_0 _S459 = _S363;

#line 183
    df_0 _S460 = df_neg_0(&_S459);

#line 183
    df_0 _S461 = _S366;

#line 183
    df_0 _S462 = df_neg_0(&_S461);

#line 183
    df_0 _S463 = _S369;

#line 183
    df_0 _S464 = df_neg_0(&_S463);

#line 183
    df_0 _S465 = _S372;

#line 183
    df_0 _S466 = df_neg_0(&_S465);

#line 183
    df_0 _S467 = _S288;

#line 183
    df_0 _S468 = _S454;

#line 183
    df_0 _S469 = df_add_0(&_S467, &_S468);

#line 183
    df_0 _S470 = _S246;

#line 183
    df_0 _S471 = _S456;

#line 183
    df_0 _S472 = df_add_0(&_S470, &_S471);

#line 183
    df_0 _S473 = _S249;

#line 183
    df_0 _S474 = _S458;

#line 183
    df_0 _S475 = df_add_0(&_S473, &_S474);

#line 183
    df_0 _S476 = _S252;

#line 183
    df_0 _S477 = _S460;

#line 183
    df_0 _S478 = df_add_0(&_S476, &_S477);

#line 183
    df_0 _S479 = _S255;

#line 183
    df_0 _S480 = _S462;

#line 183
    df_0 _S481 = df_add_0(&_S479, &_S480);

#line 183
    df_0 _S482 = _S402;

#line 183
    df_0 _S483 = df_neg_0(&_S482);

#line 183
    df_0 _S484 = _S378;

#line 183
    df_0 _S485 = df_neg_0(&_S484);

#line 183
    df_0 _S486 = _S381;

#line 183
    df_0 _S487 = df_neg_0(&_S486);

#line 183
    df_0 _S488 = _S384;

#line 183
    df_0 _S489 = df_neg_0(&_S488);

#line 183
    df_0 _S490 = _S387;

#line 183
    df_0 _S491 = df_neg_0(&_S490);

#line 183
    df_0 _S492 = _S390;

#line 183
    df_0 _S493 = df_neg_0(&_S492);

#line 183
    df_0 _S494 = _S393;

#line 183
    df_0 _S495 = df_neg_0(&_S494);

#line 183
    df_0 _S496 = _S291;

#line 183
    df_0 _S497 = _S483;

#line 183
    df_0 _S498 = df_add_0(&_S496, &_S497);

#line 183
    df_0 _S499 = _S267;

#line 183
    df_0 _S500 = _S485;

#line 183
    df_0 _S501 = df_add_0(&_S499, &_S500);

#line 183
    df_0 _S502 = _S270;

#line 183
    df_0 _S503 = _S487;

#line 183
    df_0 _S504 = df_add_0(&_S502, &_S503);

#line 183
    df_0 _S505 = _S273;

#line 183
    df_0 _S506 = _S489;

#line 183
    df_0 _S507 = df_add_0(&_S505, &_S506);

#line 183
    df_0 _S508 = _S276;

#line 183
    df_0 _S509 = _S491;

#line 183
    df_0 _S510 = df_add_0(&_S508, &_S509);

#line 183
    df_0 _S511 = _S428;

#line 183
    df_0 _S512 = _S428;

#line 183
    df_0 _S513 = df_mul_0(&_S511, &_S512);

#line 183
    df_0 _S514 = _S513;

#line 183
    df_0 _S515 = _S513;

#line 183
    df_0 _S516 = df_add_0(&_S514, &_S515);

#line 183
    df_0 _S517 = _S428;

#line 183
    df_0 _S518 = _S431;

#line 183
    df_0 _S519 = df_mul_0(&_S517, &_S518);

#line 183
    df_0 _S520 = _S431;

#line 183
    df_0 _S521 = _S428;

#line 183
    df_0 _S522 = df_mul_0(&_S520, &_S521);

#line 183
    df_0 _S523 = _S519;

#line 183
    df_0 _S524 = _S522;

#line 183
    df_0 _S525 = df_add_0(&_S523, &_S524);

#line 183
    df_0 _S526 = _S428;

#line 183
    df_0 _S527 = _S434;

#line 183
    df_0 _S528 = df_mul_0(&_S526, &_S527);

#line 183
    df_0 _S529 = _S434;

#line 183
    df_0 _S530 = _S428;

#line 183
    df_0 _S531 = df_mul_0(&_S529, &_S530);

#line 183
    df_0 _S532 = _S528;

#line 183
    df_0 _S533 = _S531;

#line 183
    df_0 _S534 = df_add_0(&_S532, &_S533);

#line 183
    df_0 _S535 = _S428;

#line 183
    df_0 _S536 = _S437;

#line 183
    df_0 _S537 = df_mul_0(&_S535, &_S536);

#line 183
    df_0 _S538 = _S437;

#line 183
    df_0 _S539 = _S428;

#line 183
    df_0 _S540 = df_mul_0(&_S538, &_S539);

#line 183
    df_0 _S541 = _S537;

#line 183
    df_0 _S542 = _S540;

#line 183
    df_0 _S543 = df_add_0(&_S541, &_S542);

#line 183
    df_0 _S544 = _S428;

#line 183
    df_0 _S545 = _S440;

#line 183
    df_0 _S546 = df_mul_0(&_S544, &_S545);

#line 183
    df_0 _S547 = _S440;

#line 183
    df_0 _S548 = _S428;

#line 183
    df_0 _S549 = df_mul_0(&_S547, &_S548);

#line 183
    df_0 _S550 = _S546;

#line 183
    df_0 _S551 = _S549;

#line 183
    df_0 _S552 = df_add_0(&_S550, &_S551);

#line 183
    df_0 _S553 = _S428;

#line 183
    df_0 _S554 = _S443;

#line 183
    df_0 _S555 = df_mul_0(&_S553, &_S554);

#line 183
    df_0 _S556 = _S443;

#line 183
    df_0 _S557 = _S428;

#line 183
    df_0 _S558 = df_mul_0(&_S556, &_S557);

#line 183
    df_0 _S559 = _S555;

#line 183
    df_0 _S560 = _S558;

#line 183
    df_0 _S561 = df_add_0(&_S559, &_S560);

#line 183
    df_0 _S562 = _S428;

#line 183
    df_0 _S563 = c1_0;

#line 183
    df_0 _S564 = df_mul_0(&_S562, &_S563);

#line 183
    df_0 _S565 = c1_0;

#line 183
    df_0 _S566 = _S428;

#line 183
    df_0 _S567 = df_mul_0(&_S565, &_S566);

#line 183
    df_0 _S568 = _S564;

#line 183
    df_0 _S569 = _S567;

#line 183
    df_0 _S570 = df_add_0(&_S568, &_S569);

#line 183
    df_0 _S571 = _S428;

#line 183
    df_0 _S572 = _S232;

#line 183
    df_0 _S573 = df_mul_0(&_S571, &_S572);

#line 183
    df_0 _S574 = _S232;

#line 183
    df_0 _S575 = _S428;

#line 183
    df_0 _S576 = df_mul_0(&_S574, &_S575);

#line 183
    df_0 _S577 = _S573;

#line 183
    df_0 _S578 = _S576;

#line 183
    df_0 _S579 = df_add_0(&_S577, &_S578);

#line 183
    df_0 _S580 = _S428;

#line 183
    df_0 _S581 = _S235;

#line 183
    df_0 _S582 = df_mul_0(&_S580, &_S581);

#line 183
    df_0 _S583 = _S235;

#line 183
    df_0 _S584 = _S428;

#line 183
    df_0 _S585 = df_mul_0(&_S583, &_S584);

#line 183
    df_0 _S586 = _S582;

#line 183
    df_0 _S587 = _S585;

#line 183
    df_0 _S588 = df_add_0(&_S586, &_S587);

#line 183
    df_0 _S589 = _S428;

#line 183
    df_0 _S590 = _S296;

#line 183
    df_0 _S591 = df_mul_0(&_S589, &_S590);

#line 183
    df_0 _S592 = _S296;

#line 183
    df_0 _S593 = _S428;

#line 183
    df_0 _S594 = df_mul_0(&_S592, &_S593);

#line 183
    df_0 _S595 = _S591;

#line 183
    df_0 _S596 = _S594;

#line 183
    df_0 _S597 = df_add_0(&_S595, &_S596);

#line 183
    df_0 _S598 = _S428;

#line 183
    df_0 _S599 = _S416;

#line 183
    df_0 _S600 = df_mul_0(&_S598, &_S599);

#line 183
    df_0 _S601 = _S416;

#line 183
    df_0 _S602 = _S428;

#line 183
    df_0 _S603 = df_mul_0(&_S601, &_S602);

#line 183
    df_0 _S604 = _S600;

#line 183
    df_0 _S605 = _S603;

#line 183
    df_0 _S606 = df_add_0(&_S604, &_S605);

#line 183
    df_0 _S607 = _S428;

#line 183
    df_0 _S608 = _S418;

#line 183
    df_0 _S609 = df_mul_0(&_S607, &_S608);

#line 183
    df_0 _S610 = _S418;

#line 183
    df_0 _S611 = _S428;

#line 183
    df_0 _S612 = df_mul_0(&_S610, &_S611);

#line 183
    df_0 _S613 = _S609;

#line 183
    df_0 _S614 = _S612;

#line 183
    df_0 _S615 = df_add_0(&_S613, &_S614);

#line 183
    df_0 _S616 = _S431;

#line 183
    df_0 _S617 = _S431;

#line 183
    df_0 _S618 = df_mul_0(&_S616, &_S617);

#line 183
    df_0 _S619 = _S618;

#line 183
    df_0 _S620 = _S618;

#line 183
    df_0 _S621 = df_add_0(&_S619, &_S620);

#line 183
    df_0 _S622 = _S446;

#line 183
    df_0 _S623 = _S425;

#line 183
    df_0 _S624 = df_mul_0(&_S622, &_S623);

#line 183
    df_0 _S625 = _S431;

#line 183
    df_0 _S626 = _S434;

#line 183
    df_0 _S627 = df_mul_0(&_S625, &_S626);

#line 183
    df_0 _S628 = _S434;

#line 183
    df_0 _S629 = _S431;

#line 183
    df_0 _S630 = df_mul_0(&_S628, &_S629);

#line 183
    df_0 _S631 = _S425;

#line 183
    df_0 _S632 = _S446;

#line 183
    df_0 _S633 = df_mul_0(&_S631, &_S632);

#line 183
    df_0 _S634 = _S624;

#line 183
    df_0 _S635 = _S627;

#line 183
    df_0 _S636 = df_add_0(&_S634, &_S635);

#line 183
    df_0 _S637 = _S636;

#line 183
    df_0 _S638 = _S630;

#line 183
    df_0 _S639 = df_add_0(&_S637, &_S638);

#line 183
    df_0 _S640 = _S639;

#line 183
    df_0 _S641 = _S633;

#line 183
    df_0 _S642 = df_add_0(&_S640, &_S641);

#line 183
    df_0 _S643 = _S431;

#line 183
    df_0 _S644 = _S437;

#line 183
    df_0 _S645 = df_mul_0(&_S643, &_S644);

#line 183
    df_0 _S646 = _S437;

#line 183
    df_0 _S647 = _S431;

#line 183
    df_0 _S648 = df_mul_0(&_S646, &_S647);

#line 183
    df_0 _S649 = _S645;

#line 183
    df_0 _S650 = _S648;

#line 183
    df_0 _S651 = df_add_0(&_S649, &_S650);

#line 183
    df_0 _S652 = _S431;

#line 183
    df_0 _S653 = _S440;

#line 183
    df_0 _S654 = df_mul_0(&_S652, &_S653);

#line 183
    df_0 _S655 = _S440;

#line 183
    df_0 _S656 = _S431;

#line 183
    df_0 _S657 = df_mul_0(&_S655, &_S656);

#line 183
    df_0 _S658 = _S654;

#line 183
    df_0 _S659 = _S657;

#line 183
    df_0 _S660 = df_add_0(&_S658, &_S659);

#line 183
    df_0 _S661 = _S449;

#line 183
    df_0 _S662 = _S425;

#line 183
    df_0 _S663 = df_mul_0(&_S661, &_S662);

#line 183
    df_0 _S664 = _S431;

#line 183
    df_0 _S665 = _S443;

#line 183
    df_0 _S666 = df_mul_0(&_S664, &_S665);

#line 183
    df_0 _S667 = _S443;

#line 183
    df_0 _S668 = _S431;

#line 183
    df_0 _S669 = df_mul_0(&_S667, &_S668);

#line 183
    df_0 _S670 = _S425;

#line 183
    df_0 _S671 = _S449;

#line 183
    df_0 _S672 = df_mul_0(&_S670, &_S671);

#line 183
    df_0 _S673 = _S663;

#line 183
    df_0 _S674 = _S666;

#line 183
    df_0 _S675 = df_add_0(&_S673, &_S674);

#line 183
    df_0 _S676 = _S675;

#line 183
    df_0 _S677 = _S669;

#line 183
    df_0 _S678 = df_add_0(&_S676, &_S677);

#line 183
    df_0 _S679 = _S678;

#line 183
    df_0 _S680 = _S672;

#line 183
    df_0 _S681 = df_add_0(&_S679, &_S680);

#line 183
    df_0 _S682 = _S431;

#line 183
    df_0 _S683 = c1_0;

#line 183
    df_0 _S684 = df_mul_0(&_S682, &_S683);

#line 183
    df_0 _S685 = c1_0;

#line 183
    df_0 _S686 = _S431;

#line 183
    df_0 _S687 = df_mul_0(&_S685, &_S686);

#line 183
    df_0 _S688 = _S684;

#line 183
    df_0 _S689 = _S687;

#line 183
    df_0 _S690 = df_add_0(&_S688, &_S689);

#line 183
    df_0 _S691 = _S431;

#line 183
    df_0 _S692 = _S232;

#line 183
    df_0 _S693 = df_mul_0(&_S691, &_S692);

#line 183
    df_0 _S694 = _S232;

#line 183
    df_0 _S695 = _S431;

#line 183
    df_0 _S696 = df_mul_0(&_S694, &_S695);

#line 183
    df_0 _S697 = _S693;

#line 183
    df_0 _S698 = _S696;

#line 183
    df_0 _S699 = df_add_0(&_S697, &_S698);

#line 183
    df_0 _S700 = _S240;

#line 183
    df_0 _S701 = _S425;

#line 183
    df_0 _S702 = df_mul_0(&_S700, &_S701);

#line 183
    df_0 _S703 = _S431;

#line 183
    df_0 _S704 = _S235;

#line 183
    df_0 _S705 = df_mul_0(&_S703, &_S704);

#line 183
    df_0 _S706 = _S235;

#line 183
    df_0 _S707 = _S431;

#line 183
    df_0 _S708 = df_mul_0(&_S706, &_S707);

#line 183
    df_0 _S709 = _S425;

#line 183
    df_0 _S710 = _S240;

#line 183
    df_0 _S711 = df_mul_0(&_S709, &_S710);

#line 183
    df_0 _S712 = _S702;

#line 183
    df_0 _S713 = _S705;

#line 183
    df_0 _S714 = df_add_0(&_S712, &_S713);

#line 183
    df_0 _S715 = _S714;

#line 183
    df_0 _S716 = _S708;

#line 183
    df_0 _S717 = df_add_0(&_S715, &_S716);

#line 183
    df_0 _S718 = _S717;

#line 183
    df_0 _S719 = _S711;

#line 183
    df_0 _S720 = df_add_0(&_S718, &_S719);

#line 183
    df_0 _S721 = _S431;

#line 183
    df_0 _S722 = _S296;

#line 183
    df_0 _S723 = df_mul_0(&_S721, &_S722);

#line 183
    df_0 _S724 = _S296;

#line 183
    df_0 _S725 = _S431;

#line 183
    df_0 _S726 = df_mul_0(&_S724, &_S725);

#line 183
    df_0 _S727 = _S723;

#line 183
    df_0 _S728 = _S726;

#line 183
    df_0 _S729 = df_add_0(&_S727, &_S728);

#line 183
    df_0 _S730 = _S431;

#line 183
    df_0 _S731 = _S416;

#line 183
    df_0 _S732 = df_mul_0(&_S730, &_S731);

#line 183
    df_0 _S733 = _S416;

#line 183
    df_0 _S734 = _S431;

#line 183
    df_0 _S735 = df_mul_0(&_S733, &_S734);

#line 183
    df_0 _S736 = _S732;

#line 183
    df_0 _S737 = _S735;

#line 183
    df_0 _S738 = df_add_0(&_S736, &_S737);

#line 183
    df_0 _S739 = _S351;

#line 183
    df_0 _S740 = _S425;

#line 183
    df_0 _S741 = df_mul_0(&_S739, &_S740);

#line 183
    df_0 _S742 = _S431;

#line 183
    df_0 _S743 = _S418;

#line 183
    df_0 _S744 = df_mul_0(&_S742, &_S743);

#line 183
    df_0 _S745 = _S418;

#line 183
    df_0 _S746 = _S431;

#line 183
    df_0 _S747 = df_mul_0(&_S745, &_S746);

#line 183
    df_0 _S748 = _S425;

#line 183
    df_0 _S749 = _S351;

#line 183
    df_0 _S750 = df_mul_0(&_S748, &_S749);

#line 183
    df_0 _S751 = _S741;

#line 183
    df_0 _S752 = _S744;

#line 183
    df_0 _S753 = df_add_0(&_S751, &_S752);

#line 183
    df_0 _S754 = _S753;

#line 183
    df_0 _S755 = _S747;

#line 183
    df_0 _S756 = df_add_0(&_S754, &_S755);

#line 183
    df_0 _S757 = _S756;

#line 183
    df_0 _S758 = _S750;

#line 183
    df_0 _S759 = df_add_0(&_S757, &_S758);

#line 183
    df_0 _S760 = _S434;

#line 183
    df_0 _S761 = _S434;

#line 183
    df_0 _S762 = df_mul_0(&_S760, &_S761);

#line 183
    df_0 _S763 = _S762;

#line 183
    df_0 _S764 = _S762;

#line 183
    df_0 _S765 = df_add_0(&_S763, &_S764);

#line 183
    df_0 _S766 = _S434;

#line 183
    df_0 _S767 = _S437;

#line 183
    df_0 _S768 = df_mul_0(&_S766, &_S767);

#line 183
    df_0 _S769 = _S437;

#line 183
    df_0 _S770 = _S434;

#line 183
    df_0 _S771 = df_mul_0(&_S769, &_S770);

#line 183
    df_0 _S772 = _S768;

#line 183
    df_0 _S773 = _S771;

#line 183
    df_0 _S774 = df_add_0(&_S772, &_S773);

#line 183
    df_0 _S775 = _S452;

#line 183
    df_0 _S776 = _S425;

#line 183
    df_0 _S777 = df_mul_0(&_S775, &_S776);

#line 183
    df_0 _S778 = _S434;

#line 183
    df_0 _S779 = _S440;

#line 183
    df_0 _S780 = df_mul_0(&_S778, &_S779);

#line 183
    df_0 _S781 = _S440;

#line 183
    df_0 _S782 = _S434;

#line 183
    df_0 _S783 = df_mul_0(&_S781, &_S782);

#line 183
    df_0 _S784 = _S425;

#line 183
    df_0 _S785 = _S452;

#line 183
    df_0 _S786 = df_mul_0(&_S784, &_S785);

#line 183
    df_0 _S787 = _S777;

#line 183
    df_0 _S788 = _S780;

#line 183
    df_0 _S789 = df_add_0(&_S787, &_S788);

#line 183
    df_0 _S790 = _S789;

#line 183
    df_0 _S791 = _S783;

#line 183
    df_0 _S792 = df_add_0(&_S790, &_S791);

#line 183
    df_0 _S793 = _S792;

#line 183
    df_0 _S794 = _S786;

#line 183
    df_0 _S795 = df_add_0(&_S793, &_S794);

#line 183
    df_0 _S796 = _S434;

#line 183
    df_0 _S797 = _S443;

#line 183
    df_0 _S798 = df_mul_0(&_S796, &_S797);

#line 183
    df_0 _S799 = _S443;

#line 183
    df_0 _S800 = _S434;

#line 183
    df_0 _S801 = df_mul_0(&_S799, &_S800);

#line 183
    df_0 _S802 = _S798;

#line 183
    df_0 _S803 = _S801;

#line 183
    df_0 _S804 = df_add_0(&_S802, &_S803);

#line 183
    df_0 _S805 = _S434;

#line 183
    df_0 _S806 = c1_0;

#line 183
    df_0 _S807 = df_mul_0(&_S805, &_S806);

#line 183
    df_0 _S808 = c1_0;

#line 183
    df_0 _S809 = _S434;

#line 183
    df_0 _S810 = df_mul_0(&_S808, &_S809);

#line 183
    df_0 _S811 = _S807;

#line 183
    df_0 _S812 = _S810;

#line 183
    df_0 _S813 = df_add_0(&_S811, &_S812);

#line 183
    df_0 _S814 = c2_0;

#line 183
    df_0 _S815 = _S425;

#line 183
    df_0 _S816 = df_mul_0(&_S814, &_S815);

#line 183
    df_0 _S817 = _S434;

#line 183
    df_0 _S818 = _S232;

#line 183
    df_0 _S819 = df_mul_0(&_S817, &_S818);

#line 183
    df_0 _S820 = _S232;

#line 183
    df_0 _S821 = _S434;

#line 183
    df_0 _S822 = df_mul_0(&_S820, &_S821);

#line 183
    df_0 _S823 = _S425;

#line 183
    df_0 _S824 = c2_0;

#line 183
    df_0 _S825 = df_mul_0(&_S823, &_S824);

#line 183
    df_0 _S826 = _S816;

#line 183
    df_0 _S827 = _S819;

#line 183
    df_0 _S828 = df_add_0(&_S826, &_S827);

#line 183
    df_0 _S829 = _S828;

#line 183
    df_0 _S830 = _S822;

#line 183
    df_0 _S831 = df_add_0(&_S829, &_S830);

#line 183
    df_0 _S832 = _S831;

#line 183
    df_0 _S833 = _S825;

#line 183
    df_0 _S834 = df_add_0(&_S832, &_S833);

#line 183
    df_0 _S835 = _S434;

#line 183
    df_0 _S836 = _S235;

#line 183
    df_0 _S837 = df_mul_0(&_S835, &_S836);

#line 183
    df_0 _S838 = _S235;

#line 183
    df_0 _S839 = _S434;

#line 183
    df_0 _S840 = df_mul_0(&_S838, &_S839);

#line 183
    df_0 _S841 = _S837;

#line 183
    df_0 _S842 = _S840;

#line 183
    df_0 _S843 = df_add_0(&_S841, &_S842);

#line 183
    df_0 _S844 = _S434;

#line 183
    df_0 _S845 = _S296;

#line 183
    df_0 _S846 = df_mul_0(&_S844, &_S845);

#line 183
    df_0 _S847 = _S296;

#line 183
    df_0 _S848 = _S434;

#line 183
    df_0 _S849 = df_mul_0(&_S847, &_S848);

#line 183
    df_0 _S850 = _S846;

#line 183
    df_0 _S851 = _S849;

#line 183
    df_0 _S852 = df_add_0(&_S850, &_S851);

#line 183
    df_0 _S853 = _S422;

#line 183
    df_0 _S854 = _S425;

#line 183
    df_0 _S855 = df_mul_0(&_S853, &_S854);

#line 183
    df_0 _S856 = _S434;

#line 183
    df_0 _S857 = _S416;

#line 183
    df_0 _S858 = df_mul_0(&_S856, &_S857);

#line 183
    df_0 _S859 = _S416;

#line 183
    df_0 _S860 = _S434;

#line 183
    df_0 _S861 = df_mul_0(&_S859, &_S860);

#line 183
    df_0 _S862 = _S425;

#line 183
    df_0 _S863 = _S422;

#line 183
    df_0 _S864 = df_mul_0(&_S862, &_S863);

#line 183
    df_0 _S865 = _S855;

#line 183
    df_0 _S866 = _S858;

#line 183
    df_0 _S867 = df_add_0(&_S865, &_S866);

#line 183
    df_0 _S868 = _S867;

#line 183
    df_0 _S869 = _S861;

#line 183
    df_0 _S870 = df_add_0(&_S868, &_S869);

#line 183
    df_0 _S871 = _S870;

#line 183
    df_0 _S872 = _S864;

#line 183
    df_0 _S873 = df_add_0(&_S871, &_S872);

#line 183
    df_0 _S874 = _S434;

#line 183
    df_0 _S875 = _S418;

#line 183
    df_0 _S876 = df_mul_0(&_S874, &_S875);

#line 183
    df_0 _S877 = _S418;

#line 183
    df_0 _S878 = _S434;

#line 183
    df_0 _S879 = df_mul_0(&_S877, &_S878);

#line 183
    df_0 _S880 = _S876;

#line 183
    df_0 _S881 = _S879;

#line 183
    df_0 _S882 = df_add_0(&_S880, &_S881);

#line 183
    df_0 _S883 = _S437;

#line 183
    df_0 _S884 = _S437;

#line 183
    df_0 _S885 = df_mul_0(&_S883, &_S884);

#line 183
    df_0 _S886 = _S885;

#line 183
    df_0 _S887 = _S885;

#line 183
    df_0 _S888 = df_add_0(&_S886, &_S887);

#line 183
    df_0 _S889 = _S437;

#line 183
    df_0 _S890 = _S440;

#line 183
    df_0 _S891 = df_mul_0(&_S889, &_S890);

#line 183
    df_0 _S892 = _S440;

#line 183
    df_0 _S893 = _S437;

#line 183
    df_0 _S894 = df_mul_0(&_S892, &_S893);

#line 183
    df_0 _S895 = _S891;

#line 183
    df_0 _S896 = _S894;

#line 183
    df_0 _S897 = df_add_0(&_S895, &_S896);

#line 183
    df_0 _S898 = _S437;

#line 183
    df_0 _S899 = _S443;

#line 183
    df_0 _S900 = df_mul_0(&_S898, &_S899);

#line 183
    df_0 _S901 = _S443;

#line 183
    df_0 _S902 = _S437;

#line 183
    df_0 _S903 = df_mul_0(&_S901, &_S902);

#line 183
    df_0 _S904 = _S900;

#line 183
    df_0 _S905 = _S903;

#line 183
    df_0 _S906 = df_add_0(&_S904, &_S905);

#line 183
    df_0 _S907 = _S437;

#line 183
    df_0 _S908 = c1_0;

#line 183
    df_0 _S909 = df_mul_0(&_S907, &_S908);

#line 183
    df_0 _S910 = c1_0;

#line 183
    df_0 _S911 = _S437;

#line 183
    df_0 _S912 = df_mul_0(&_S910, &_S911);

#line 183
    df_0 _S913 = _S909;

#line 183
    df_0 _S914 = _S912;

#line 183
    df_0 _S915 = df_add_0(&_S913, &_S914);

#line 183
    df_0 _S916 = _S437;

#line 183
    df_0 _S917 = _S232;

#line 183
    df_0 _S918 = df_mul_0(&_S916, &_S917);

#line 183
    df_0 _S919 = _S232;

#line 183
    df_0 _S920 = _S437;

#line 183
    df_0 _S921 = df_mul_0(&_S919, &_S920);

#line 183
    df_0 _S922 = _S918;

#line 183
    df_0 _S923 = _S921;

#line 183
    df_0 _S924 = df_add_0(&_S922, &_S923);

#line 183
    df_0 _S925 = _S437;

#line 183
    df_0 _S926 = _S235;

#line 183
    df_0 _S927 = df_mul_0(&_S925, &_S926);

#line 183
    df_0 _S928 = _S235;

#line 183
    df_0 _S929 = _S437;

#line 183
    df_0 _S930 = df_mul_0(&_S928, &_S929);

#line 183
    df_0 _S931 = _S927;

#line 183
    df_0 _S932 = _S930;

#line 183
    df_0 _S933 = df_add_0(&_S931, &_S932);

#line 183
    df_0 _S934 = _S437;

#line 183
    df_0 _S935 = _S296;

#line 183
    df_0 _S936 = df_mul_0(&_S934, &_S935);

#line 183
    df_0 _S937 = _S296;

#line 183
    df_0 _S938 = _S437;

#line 183
    df_0 _S939 = df_mul_0(&_S937, &_S938);

#line 183
    df_0 _S940 = _S936;

#line 183
    df_0 _S941 = _S939;

#line 183
    df_0 _S942 = df_add_0(&_S940, &_S941);

#line 183
    df_0 _S943 = _S437;

#line 183
    df_0 _S944 = _S416;

#line 183
    df_0 _S945 = df_mul_0(&_S943, &_S944);

#line 183
    df_0 _S946 = _S416;

#line 183
    df_0 _S947 = _S437;

#line 183
    df_0 _S948 = df_mul_0(&_S946, &_S947);

#line 183
    df_0 _S949 = _S945;

#line 183
    df_0 _S950 = _S948;

#line 183
    df_0 _S951 = df_add_0(&_S949, &_S950);

#line 183
    df_0 _S952 = _S437;

#line 183
    df_0 _S953 = _S418;

#line 183
    df_0 _S954 = df_mul_0(&_S952, &_S953);

#line 183
    df_0 _S955 = _S418;

#line 183
    df_0 _S956 = _S437;

#line 183
    df_0 _S957 = df_mul_0(&_S955, &_S956);

#line 183
    df_0 _S958 = _S954;

#line 183
    df_0 _S959 = _S957;

#line 183
    df_0 _S960 = df_add_0(&_S958, &_S959);

#line 183
    df_0 _S961 = _S440;

#line 183
    df_0 _S962 = _S440;

#line 183
    df_0 _S963 = df_mul_0(&_S961, &_S962);

#line 183
    df_0 _S964 = _S963;

#line 183
    df_0 _S965 = _S963;

#line 183
    df_0 _S966 = df_add_0(&_S964, &_S965);

#line 183
    df_0 _S967 = _S440;

#line 183
    df_0 _S968 = _S443;

#line 183
    df_0 _S969 = df_mul_0(&_S967, &_S968);

#line 183
    df_0 _S970 = _S443;

#line 183
    df_0 _S971 = _S440;

#line 183
    df_0 _S972 = df_mul_0(&_S970, &_S971);

#line 183
    df_0 _S973 = _S969;

#line 183
    df_0 _S974 = _S972;

#line 183
    df_0 _S975 = df_add_0(&_S973, &_S974);

#line 183
    df_0 _S976 = _S440;

#line 183
    df_0 _S977 = c1_0;

#line 183
    df_0 _S978 = df_mul_0(&_S976, &_S977);

#line 183
    df_0 _S979 = c1_0;

#line 183
    df_0 _S980 = _S440;

#line 183
    df_0 _S981 = df_mul_0(&_S979, &_S980);

#line 183
    df_0 _S982 = _S978;

#line 183
    df_0 _S983 = _S981;

#line 183
    df_0 _S984 = df_add_0(&_S982, &_S983);

#line 183
    df_0 _S985 = _S440;

#line 183
    df_0 _S986 = _S232;

#line 183
    df_0 _S987 = df_mul_0(&_S985, &_S986);

#line 183
    df_0 _S988 = _S232;

#line 183
    df_0 _S989 = _S440;

#line 183
    df_0 _S990 = df_mul_0(&_S988, &_S989);

#line 183
    df_0 _S991 = _S987;

#line 183
    df_0 _S992 = _S990;

#line 183
    df_0 _S993 = df_add_0(&_S991, &_S992);

#line 183
    df_0 _S994 = _S440;

#line 183
    df_0 _S995 = _S235;

#line 183
    df_0 _S996 = df_mul_0(&_S994, &_S995);

#line 183
    df_0 _S997 = _S235;

#line 183
    df_0 _S998 = _S440;

#line 183
    df_0 _S999 = df_mul_0(&_S997, &_S998);

#line 183
    df_0 _S1000 = _S816;

#line 183
    df_0 _S1001 = _S996;

#line 183
    df_0 _S1002 = df_add_0(&_S1000, &_S1001);

#line 183
    df_0 _S1003 = _S1002;

#line 183
    df_0 _S1004 = _S999;

#line 183
    df_0 _S1005 = df_add_0(&_S1003, &_S1004);

#line 183
    df_0 _S1006 = _S1005;

#line 183
    df_0 _S1007 = _S825;

#line 183
    df_0 _S1008 = df_add_0(&_S1006, &_S1007);

#line 183
    df_0 _S1009 = _S440;

#line 183
    df_0 _S1010 = _S296;

#line 183
    df_0 _S1011 = df_mul_0(&_S1009, &_S1010);

#line 183
    df_0 _S1012 = _S296;

#line 183
    df_0 _S1013 = _S440;

#line 183
    df_0 _S1014 = df_mul_0(&_S1012, &_S1013);

#line 183
    df_0 _S1015 = _S1011;

#line 183
    df_0 _S1016 = _S1014;

#line 183
    df_0 _S1017 = df_add_0(&_S1015, &_S1016);

#line 183
    df_0 _S1018 = _S440;

#line 183
    df_0 _S1019 = _S416;

#line 183
    df_0 _S1020 = df_mul_0(&_S1018, &_S1019);

#line 183
    df_0 _S1021 = _S416;

#line 183
    df_0 _S1022 = _S440;

#line 183
    df_0 _S1023 = df_mul_0(&_S1021, &_S1022);

#line 183
    df_0 _S1024 = _S1020;

#line 183
    df_0 _S1025 = _S1023;

#line 183
    df_0 _S1026 = df_add_0(&_S1024, &_S1025);

#line 183
    df_0 _S1027 = _S440;

#line 183
    df_0 _S1028 = _S418;

#line 183
    df_0 _S1029 = df_mul_0(&_S1027, &_S1028);

#line 183
    df_0 _S1030 = _S418;

#line 183
    df_0 _S1031 = _S440;

#line 183
    df_0 _S1032 = df_mul_0(&_S1030, &_S1031);

#line 183
    df_0 _S1033 = _S855;

#line 183
    df_0 _S1034 = _S1029;

#line 183
    df_0 _S1035 = df_add_0(&_S1033, &_S1034);

#line 183
    df_0 _S1036 = _S1035;

#line 183
    df_0 _S1037 = _S1032;

#line 183
    df_0 _S1038 = df_add_0(&_S1036, &_S1037);

#line 183
    df_0 _S1039 = _S1038;

#line 183
    df_0 _S1040 = _S864;

#line 183
    df_0 _S1041 = df_add_0(&_S1039, &_S1040);

#line 183
    df_0 _S1042 = _S443;

#line 183
    df_0 _S1043 = _S443;

#line 183
    df_0 _S1044 = df_mul_0(&_S1042, &_S1043);

#line 183
    df_0 _S1045 = _S1044;

#line 183
    df_0 _S1046 = _S1044;

#line 183
    df_0 _S1047 = df_add_0(&_S1045, &_S1046);

#line 183
    df_0 _S1048 = _S443;

#line 183
    df_0 _S1049 = c1_0;

#line 183
    df_0 _S1050 = df_mul_0(&_S1048, &_S1049);

#line 183
    df_0 _S1051 = c1_0;

#line 183
    df_0 _S1052 = _S443;

#line 183
    df_0 _S1053 = df_mul_0(&_S1051, &_S1052);

#line 183
    df_0 _S1054 = _S1050;

#line 183
    df_0 _S1055 = _S1053;

#line 183
    df_0 _S1056 = df_add_0(&_S1054, &_S1055);

#line 183
    df_0 _S1057 = _S443;

#line 183
    df_0 _S1058 = _S232;

#line 183
    df_0 _S1059 = df_mul_0(&_S1057, &_S1058);

#line 183
    df_0 _S1060 = _S232;

#line 183
    df_0 _S1061 = _S443;

#line 183
    df_0 _S1062 = df_mul_0(&_S1060, &_S1061);

#line 183
    df_0 _S1063 = _S702;

#line 183
    df_0 _S1064 = _S1059;

#line 183
    df_0 _S1065 = df_add_0(&_S1063, &_S1064);

#line 183
    df_0 _S1066 = _S1065;

#line 183
    df_0 _S1067 = _S1062;

#line 183
    df_0 _S1068 = df_add_0(&_S1066, &_S1067);

#line 183
    df_0 _S1069 = _S1068;

#line 183
    df_0 _S1070 = _S711;

#line 183
    df_0 _S1071 = df_add_0(&_S1069, &_S1070);

#line 183
    df_0 _S1072 = _S443;

#line 183
    df_0 _S1073 = _S235;

#line 183
    df_0 _S1074 = df_mul_0(&_S1072, &_S1073);

#line 183
    df_0 _S1075 = _S235;

#line 183
    df_0 _S1076 = _S443;

#line 183
    df_0 _S1077 = df_mul_0(&_S1075, &_S1076);

#line 183
    df_0 _S1078 = _S1074;

#line 183
    df_0 _S1079 = _S1077;

#line 183
    df_0 _S1080 = df_add_0(&_S1078, &_S1079);

#line 183
    df_0 _S1081 = _S443;

#line 183
    df_0 _S1082 = _S296;

#line 183
    df_0 _S1083 = df_mul_0(&_S1081, &_S1082);

#line 183
    df_0 _S1084 = _S296;

#line 183
    df_0 _S1085 = _S443;

#line 183
    df_0 _S1086 = df_mul_0(&_S1084, &_S1085);

#line 183
    df_0 _S1087 = _S1083;

#line 183
    df_0 _S1088 = _S1086;

#line 183
    df_0 _S1089 = df_add_0(&_S1087, &_S1088);

#line 183
    df_0 _S1090 = _S443;

#line 183
    df_0 _S1091 = _S416;

#line 183
    df_0 _S1092 = df_mul_0(&_S1090, &_S1091);

#line 183
    df_0 _S1093 = _S416;

#line 183
    df_0 _S1094 = _S443;

#line 183
    df_0 _S1095 = df_mul_0(&_S1093, &_S1094);

#line 183
    df_0 _S1096 = _S741;

#line 183
    df_0 _S1097 = _S1092;

#line 183
    df_0 _S1098 = df_add_0(&_S1096, &_S1097);

#line 183
    df_0 _S1099 = _S1098;

#line 183
    df_0 _S1100 = _S1095;

#line 183
    df_0 _S1101 = df_add_0(&_S1099, &_S1100);

#line 183
    df_0 _S1102 = _S1101;

#line 183
    df_0 _S1103 = _S750;

#line 183
    df_0 _S1104 = df_add_0(&_S1102, &_S1103);

#line 183
    df_0 _S1105 = _S443;

#line 183
    df_0 _S1106 = _S418;

#line 183
    df_0 _S1107 = df_mul_0(&_S1105, &_S1106);

#line 183
    df_0 _S1108 = _S418;

#line 183
    df_0 _S1109 = _S443;

#line 183
    df_0 _S1110 = df_mul_0(&_S1108, &_S1109);

#line 183
    df_0 _S1111 = _S1107;

#line 183
    df_0 _S1112 = _S1110;

#line 183
    df_0 _S1113 = df_add_0(&_S1111, &_S1112);

#line 183
    df_0 _S1114 = c1_0;

#line 183
    df_0 _S1115 = c1_0;

#line 183
    df_0 _S1116 = df_mul_0(&_S1114, &_S1115);

#line 183
    df_0 _S1117 = _S1116;

#line 183
    df_0 _S1118 = _S1116;

#line 183
    df_0 _S1119 = df_add_0(&_S1117, &_S1118);

#line 183
    df_0 _S1120 = c1_0;

#line 183
    df_0 _S1121 = _S232;

#line 183
    df_0 _S1122 = df_mul_0(&_S1120, &_S1121);

#line 183
    df_0 _S1123 = _S232;

#line 183
    df_0 _S1124 = c1_0;

#line 183
    df_0 _S1125 = df_mul_0(&_S1123, &_S1124);

#line 183
    df_0 _S1126 = _S1122;

#line 183
    df_0 _S1127 = _S1125;

#line 183
    df_0 _S1128 = df_add_0(&_S1126, &_S1127);

#line 183
    df_0 _S1129 = c1_0;

#line 183
    df_0 _S1130 = _S235;

#line 183
    df_0 _S1131 = df_mul_0(&_S1129, &_S1130);

#line 183
    df_0 _S1132 = _S235;

#line 183
    df_0 _S1133 = c1_0;

#line 183
    df_0 _S1134 = df_mul_0(&_S1132, &_S1133);

#line 183
    df_0 _S1135 = _S1131;

#line 183
    df_0 _S1136 = _S1134;

#line 183
    df_0 _S1137 = df_add_0(&_S1135, &_S1136);

#line 183
    df_0 _S1138 = c1_0;

#line 183
    df_0 _S1139 = _S296;

#line 183
    df_0 _S1140 = df_mul_0(&_S1138, &_S1139);

#line 183
    df_0 _S1141 = _S296;

#line 183
    df_0 _S1142 = c1_0;

#line 183
    df_0 _S1143 = df_mul_0(&_S1141, &_S1142);

#line 183
    df_0 _S1144 = _S1140;

#line 183
    df_0 _S1145 = _S1143;

#line 183
    df_0 _S1146 = df_add_0(&_S1144, &_S1145);

#line 183
    df_0 _S1147 = c1_0;

#line 183
    df_0 _S1148 = _S416;

#line 183
    df_0 _S1149 = df_mul_0(&_S1147, &_S1148);

#line 183
    df_0 _S1150 = _S416;

#line 183
    df_0 _S1151 = c1_0;

#line 183
    df_0 _S1152 = df_mul_0(&_S1150, &_S1151);

#line 183
    df_0 _S1153 = _S1149;

#line 183
    df_0 _S1154 = _S1152;

#line 183
    df_0 _S1155 = df_add_0(&_S1153, &_S1154);

#line 183
    df_0 _S1156 = c1_0;

#line 183
    df_0 _S1157 = _S418;

#line 183
    df_0 _S1158 = df_mul_0(&_S1156, &_S1157);

#line 183
    df_0 _S1159 = _S418;

#line 183
    df_0 _S1160 = c1_0;

#line 183
    df_0 _S1161 = df_mul_0(&_S1159, &_S1160);

#line 183
    df_0 _S1162 = _S1158;

#line 183
    df_0 _S1163 = _S1161;

#line 183
    df_0 _S1164 = df_add_0(&_S1162, &_S1163);

#line 183
    df_0 _S1165 = _S232;

#line 183
    df_0 _S1166 = _S232;

#line 183
    df_0 _S1167 = df_mul_0(&_S1165, &_S1166);

#line 183
    df_0 _S1168 = _S1167;

#line 183
    df_0 _S1169 = _S1167;

#line 183
    df_0 _S1170 = df_add_0(&_S1168, &_S1169);

#line 183
    df_0 _S1171 = _S232;

#line 183
    df_0 _S1172 = _S235;

#line 183
    df_0 _S1173 = df_mul_0(&_S1171, &_S1172);

#line 183
    df_0 _S1174 = _S235;

#line 183
    df_0 _S1175 = _S232;

#line 183
    df_0 _S1176 = df_mul_0(&_S1174, &_S1175);

#line 183
    df_0 _S1177 = _S1173;

#line 183
    df_0 _S1178 = _S1176;

#line 183
    df_0 _S1179 = df_add_0(&_S1177, &_S1178);

#line 183
    df_0 _S1180 = _S232;

#line 183
    df_0 _S1181 = _S296;

#line 183
    df_0 _S1182 = df_mul_0(&_S1180, &_S1181);

#line 183
    df_0 _S1183 = _S296;

#line 183
    df_0 _S1184 = _S232;

#line 183
    df_0 _S1185 = df_mul_0(&_S1183, &_S1184);

#line 183
    df_0 _S1186 = _S1182;

#line 183
    df_0 _S1187 = _S1185;

#line 183
    df_0 _S1188 = df_add_0(&_S1186, &_S1187);

#line 183
    df_0 _S1189 = _S232;

#line 183
    df_0 _S1190 = _S416;

#line 183
    df_0 _S1191 = df_mul_0(&_S1189, &_S1190);

#line 183
    df_0 _S1192 = _S416;

#line 183
    df_0 _S1193 = _S232;

#line 183
    df_0 _S1194 = df_mul_0(&_S1192, &_S1193);

#line 183
    df_0 _S1195 = _S1191;

#line 183
    df_0 _S1196 = _S1194;

#line 183
    df_0 _S1197 = df_add_0(&_S1195, &_S1196);

#line 183
    df_0 _S1198 = _S232;

#line 183
    df_0 _S1199 = _S418;

#line 183
    df_0 _S1200 = df_mul_0(&_S1198, &_S1199);

#line 183
    df_0 _S1201 = _S418;

#line 183
    df_0 _S1202 = _S232;

#line 183
    df_0 _S1203 = df_mul_0(&_S1201, &_S1202);

#line 183
    df_0 _S1204 = _S1200;

#line 183
    df_0 _S1205 = _S1203;

#line 183
    df_0 _S1206 = df_add_0(&_S1204, &_S1205);

#line 183
    df_0 _S1207 = _S235;

#line 183
    df_0 _S1208 = _S235;

#line 183
    df_0 _S1209 = df_mul_0(&_S1207, &_S1208);

#line 183
    df_0 _S1210 = _S1209;

#line 183
    df_0 _S1211 = _S1209;

#line 183
    df_0 _S1212 = df_add_0(&_S1210, &_S1211);

#line 183
    df_0 _S1213 = _S235;

#line 183
    df_0 _S1214 = _S296;

#line 183
    df_0 _S1215 = df_mul_0(&_S1213, &_S1214);

#line 183
    df_0 _S1216 = _S296;

#line 183
    df_0 _S1217 = _S235;

#line 183
    df_0 _S1218 = df_mul_0(&_S1216, &_S1217);

#line 183
    df_0 _S1219 = _S1215;

#line 183
    df_0 _S1220 = _S1218;

#line 183
    df_0 _S1221 = df_add_0(&_S1219, &_S1220);

#line 183
    df_0 _S1222 = _S235;

#line 183
    df_0 _S1223 = _S416;

#line 183
    df_0 _S1224 = df_mul_0(&_S1222, &_S1223);

#line 183
    df_0 _S1225 = _S416;

#line 183
    df_0 _S1226 = _S235;

#line 183
    df_0 _S1227 = df_mul_0(&_S1225, &_S1226);

#line 183
    df_0 _S1228 = _S1224;

#line 183
    df_0 _S1229 = _S1227;

#line 183
    df_0 _S1230 = df_add_0(&_S1228, &_S1229);

#line 183
    df_0 _S1231 = _S235;

#line 183
    df_0 _S1232 = _S418;

#line 183
    df_0 _S1233 = df_mul_0(&_S1231, &_S1232);

#line 183
    df_0 _S1234 = _S418;

#line 183
    df_0 _S1235 = _S235;

#line 183
    df_0 _S1236 = df_mul_0(&_S1234, &_S1235);

#line 183
    df_0 _S1237 = _S1233;

#line 183
    df_0 _S1238 = _S1236;

#line 183
    df_0 _S1239 = df_add_0(&_S1237, &_S1238);

#line 183
    df_0 _S1240 = _S296;

#line 183
    df_0 _S1241 = _S296;

#line 183
    df_0 _S1242 = df_mul_0(&_S1240, &_S1241);

#line 183
    df_0 _S1243 = _S1242;

#line 183
    df_0 _S1244 = _S1242;

#line 183
    df_0 _S1245 = df_add_0(&_S1243, &_S1244);

#line 183
    df_0 _S1246 = _S296;

#line 183
    df_0 _S1247 = _S416;

#line 183
    df_0 _S1248 = df_mul_0(&_S1246, &_S1247);

#line 183
    df_0 _S1249 = _S416;

#line 183
    df_0 _S1250 = _S296;

#line 183
    df_0 _S1251 = df_mul_0(&_S1249, &_S1250);

#line 183
    df_0 _S1252 = _S1248;

#line 183
    df_0 _S1253 = _S1251;

#line 183
    df_0 _S1254 = df_add_0(&_S1252, &_S1253);

#line 183
    df_0 _S1255 = _S296;

#line 183
    df_0 _S1256 = _S418;

#line 183
    df_0 _S1257 = df_mul_0(&_S1255, &_S1256);

#line 183
    df_0 _S1258 = _S418;

#line 183
    df_0 _S1259 = _S296;

#line 183
    df_0 _S1260 = df_mul_0(&_S1258, &_S1259);

#line 183
    df_0 _S1261 = _S1257;

#line 183
    df_0 _S1262 = _S1260;

#line 183
    df_0 _S1263 = df_add_0(&_S1261, &_S1262);

#line 183
    df_0 _S1264 = _S416;

#line 183
    df_0 _S1265 = _S416;

#line 183
    df_0 _S1266 = df_mul_0(&_S1264, &_S1265);

#line 183
    df_0 _S1267 = _S1266;

#line 183
    df_0 _S1268 = _S1266;

#line 183
    df_0 _S1269 = df_add_0(&_S1267, &_S1268);

#line 183
    df_0 _S1270 = _S416;

#line 183
    df_0 _S1271 = _S418;

#line 183
    df_0 _S1272 = df_mul_0(&_S1270, &_S1271);

#line 183
    df_0 _S1273 = _S418;

#line 183
    df_0 _S1274 = _S416;

#line 183
    df_0 _S1275 = df_mul_0(&_S1273, &_S1274);

#line 183
    df_0 _S1276 = _S1272;

#line 183
    df_0 _S1277 = _S1275;

#line 183
    df_0 _S1278 = df_add_0(&_S1276, &_S1277);

#line 183
    df_0 _S1279 = _S418;

#line 183
    df_0 _S1280 = _S418;

#line 183
    df_0 _S1281 = df_mul_0(&_S1279, &_S1280);

#line 183
    df_0 _S1282 = _S1281;

#line 183
    df_0 _S1283 = _S1281;

#line 183
    df_0 _S1284 = df_add_0(&_S1282, &_S1283);

#line 183
    df_0 _S1285 = _S472;

#line 183
    df_0 _S1286 = _S472;

#line 183
    df_0 _S1287 = df_mul_0(&_S1285, &_S1286);

#line 183
    df_0 _S1288 = _S1287;

#line 183
    df_0 _S1289 = _S1287;

#line 183
    df_0 _S1290 = df_add_0(&_S1288, &_S1289);

#line 183
    df_0 _S1291 = _S472;

#line 183
    df_0 _S1292 = _S428;

#line 183
    df_0 _S1293 = df_mul_0(&_S1291, &_S1292);

#line 183
    df_0 _S1294 = _S428;

#line 183
    df_0 _S1295 = _S472;

#line 183
    df_0 _S1296 = df_mul_0(&_S1294, &_S1295);

#line 183
    df_0 _S1297 = _S1293;

#line 183
    df_0 _S1298 = _S1296;

#line 183
    df_0 _S1299 = df_add_0(&_S1297, &_S1298);

#line 183
    df_0 _S1300 = _S446;

#line 183
    df_0 _S1301 = _S469;

#line 183
    df_0 _S1302 = df_mul_0(&_S1300, &_S1301);

#line 183
    df_0 _S1303 = _S472;

#line 183
    df_0 _S1304 = _S475;

#line 183
    df_0 _S1305 = df_mul_0(&_S1303, &_S1304);

#line 183
    df_0 _S1306 = _S475;

#line 183
    df_0 _S1307 = _S472;

#line 183
    df_0 _S1308 = df_mul_0(&_S1306, &_S1307);

#line 183
    df_0 _S1309 = _S469;

#line 183
    df_0 _S1310 = _S446;

#line 183
    df_0 _S1311 = df_mul_0(&_S1309, &_S1310);

#line 183
    df_0 _S1312 = _S1302;

#line 183
    df_0 _S1313 = _S1305;

#line 183
    df_0 _S1314 = df_add_0(&_S1312, &_S1313);

#line 183
    df_0 _S1315 = _S1314;

#line 183
    df_0 _S1316 = _S1308;

#line 183
    df_0 _S1317 = df_add_0(&_S1315, &_S1316);

#line 183
    df_0 _S1318 = _S1317;

#line 183
    df_0 _S1319 = _S1311;

#line 183
    df_0 _S1320 = df_add_0(&_S1318, &_S1319);

#line 183
    df_0 _S1321 = _S472;

#line 183
    df_0 _S1322 = _S478;

#line 183
    df_0 _S1323 = df_mul_0(&_S1321, &_S1322);

#line 183
    df_0 _S1324 = _S478;

#line 183
    df_0 _S1325 = _S472;

#line 183
    df_0 _S1326 = df_mul_0(&_S1324, &_S1325);

#line 183
    df_0 _S1327 = _S1323;

#line 183
    df_0 _S1328 = _S1326;

#line 183
    df_0 _S1329 = df_add_0(&_S1327, &_S1328);

#line 183
    df_0 _S1330 = _S472;

#line 183
    df_0 _S1331 = _S437;

#line 183
    df_0 _S1332 = df_mul_0(&_S1330, &_S1331);

#line 183
    df_0 _S1333 = _S437;

#line 183
    df_0 _S1334 = _S472;

#line 183
    df_0 _S1335 = df_mul_0(&_S1333, &_S1334);

#line 183
    df_0 _S1336 = _S1332;

#line 183
    df_0 _S1337 = _S1335;

#line 183
    df_0 _S1338 = df_add_0(&_S1336, &_S1337);

#line 183
    df_0 _S1339 = _S452;

#line 183
    df_0 _S1340 = _S469;

#line 183
    df_0 _S1341 = df_mul_0(&_S1339, &_S1340);

#line 183
    df_0 _S1342 = _S472;

#line 183
    df_0 _S1343 = _S481;

#line 183
    df_0 _S1344 = df_mul_0(&_S1342, &_S1343);

#line 183
    df_0 _S1345 = _S481;

#line 183
    df_0 _S1346 = _S472;

#line 183
    df_0 _S1347 = df_mul_0(&_S1345, &_S1346);

#line 183
    df_0 _S1348 = _S469;

#line 183
    df_0 _S1349 = _S452;

#line 183
    df_0 _S1350 = df_mul_0(&_S1348, &_S1349);

#line 183
    df_0 _S1351 = _S1341;

#line 183
    df_0 _S1352 = _S1344;

#line 183
    df_0 _S1353 = df_add_0(&_S1351, &_S1352);

#line 183
    df_0 _S1354 = _S1353;

#line 183
    df_0 _S1355 = _S1347;

#line 183
    df_0 _S1356 = df_add_0(&_S1354, &_S1355);

#line 183
    df_0 _S1357 = _S1356;

#line 183
    df_0 _S1358 = _S1350;

#line 183
    df_0 _S1359 = df_add_0(&_S1357, &_S1358);

#line 183
    df_0 _S1360 = _S472;

#line 183
    df_0 _S1361 = _S258;

#line 183
    df_0 _S1362 = df_mul_0(&_S1360, &_S1361);

#line 183
    df_0 _S1363 = _S258;

#line 183
    df_0 _S1364 = _S472;

#line 183
    df_0 _S1365 = df_mul_0(&_S1363, &_S1364);

#line 183
    df_0 _S1366 = _S1362;

#line 183
    df_0 _S1367 = _S1365;

#line 183
    df_0 _S1368 = df_add_0(&_S1366, &_S1367);

#line 183
    df_0 _S1369 = _S472;

#line 183
    df_0 _S1370 = c1_0;

#line 183
    df_0 _S1371 = df_mul_0(&_S1369, &_S1370);

#line 183
    df_0 _S1372 = c1_0;

#line 183
    df_0 _S1373 = _S472;

#line 183
    df_0 _S1374 = df_mul_0(&_S1372, &_S1373);

#line 183
    df_0 _S1375 = _S1371;

#line 183
    df_0 _S1376 = _S1374;

#line 183
    df_0 _S1377 = df_add_0(&_S1375, &_S1376);

#line 183
    df_0 _S1378 = c2_0;

#line 183
    df_0 _S1379 = _S469;

#line 183
    df_0 _S1380 = df_mul_0(&_S1378, &_S1379);

#line 183
    df_0 _S1381 = _S472;

#line 183
    df_0 _S1382 = _S261;

#line 183
    df_0 _S1383 = df_mul_0(&_S1381, &_S1382);

#line 183
    df_0 _S1384 = _S261;

#line 183
    df_0 _S1385 = _S472;

#line 183
    df_0 _S1386 = df_mul_0(&_S1384, &_S1385);

#line 183
    df_0 _S1387 = _S469;

#line 183
    df_0 _S1388 = c2_0;

#line 183
    df_0 _S1389 = df_mul_0(&_S1387, &_S1388);

#line 183
    df_0 _S1390 = _S1380;

#line 183
    df_0 _S1391 = _S1383;

#line 183
    df_0 _S1392 = df_add_0(&_S1390, &_S1391);

#line 183
    df_0 _S1393 = _S1392;

#line 183
    df_0 _S1394 = _S1386;

#line 183
    df_0 _S1395 = df_add_0(&_S1393, &_S1394);

#line 183
    df_0 _S1396 = _S1395;

#line 183
    df_0 _S1397 = _S1389;

#line 183
    df_0 _S1398 = df_add_0(&_S1396, &_S1397);

#line 183
    df_0 _S1399 = _S472;

#line 183
    df_0 _S1400 = _S464;

#line 183
    df_0 _S1401 = df_mul_0(&_S1399, &_S1400);

#line 183
    df_0 _S1402 = _S464;

#line 183
    df_0 _S1403 = _S472;

#line 183
    df_0 _S1404 = df_mul_0(&_S1402, &_S1403);

#line 183
    df_0 _S1405 = _S1401;

#line 183
    df_0 _S1406 = _S1404;

#line 183
    df_0 _S1407 = df_add_0(&_S1405, &_S1406);

#line 183
    df_0 _S1408 = _S472;

#line 183
    df_0 _S1409 = _S296;

#line 183
    df_0 _S1410 = df_mul_0(&_S1408, &_S1409);

#line 183
    df_0 _S1411 = _S296;

#line 183
    df_0 _S1412 = _S472;

#line 183
    df_0 _S1413 = df_mul_0(&_S1411, &_S1412);

#line 183
    df_0 _S1414 = _S1410;

#line 183
    df_0 _S1415 = _S1413;

#line 183
    df_0 _S1416 = df_add_0(&_S1414, &_S1415);

#line 183
    df_0 _S1417 = _S422;

#line 183
    df_0 _S1418 = _S469;

#line 183
    df_0 _S1419 = df_mul_0(&_S1417, &_S1418);

#line 183
    df_0 _S1420 = _S472;

#line 183
    df_0 _S1421 = _S466;

#line 183
    df_0 _S1422 = df_mul_0(&_S1420, &_S1421);

#line 183
    df_0 _S1423 = _S466;

#line 183
    df_0 _S1424 = _S472;

#line 183
    df_0 _S1425 = df_mul_0(&_S1423, &_S1424);

#line 183
    df_0 _S1426 = _S469;

#line 183
    df_0 _S1427 = _S422;

#line 183
    df_0 _S1428 = df_mul_0(&_S1426, &_S1427);

#line 183
    df_0 _S1429 = _S1419;

#line 183
    df_0 _S1430 = _S1422;

#line 183
    df_0 _S1431 = df_add_0(&_S1429, &_S1430);

#line 183
    df_0 _S1432 = _S1431;

#line 183
    df_0 _S1433 = _S1425;

#line 183
    df_0 _S1434 = df_add_0(&_S1432, &_S1433);

#line 183
    df_0 _S1435 = _S1434;

#line 183
    df_0 _S1436 = _S1428;

#line 183
    df_0 _S1437 = df_add_0(&_S1435, &_S1436);

#line 183
    df_0 _S1438 = _S428;

#line 183
    df_0 _S1439 = _S475;

#line 183
    df_0 _S1440 = df_mul_0(&_S1438, &_S1439);

#line 183
    df_0 _S1441 = _S475;

#line 183
    df_0 _S1442 = _S428;

#line 183
    df_0 _S1443 = df_mul_0(&_S1441, &_S1442);

#line 183
    df_0 _S1444 = _S1440;

#line 183
    df_0 _S1445 = _S1443;

#line 183
    df_0 _S1446 = df_add_0(&_S1444, &_S1445);

#line 183
    df_0 _S1447 = _S428;

#line 183
    df_0 _S1448 = _S478;

#line 183
    df_0 _S1449 = df_mul_0(&_S1447, &_S1448);

#line 183
    df_0 _S1450 = _S478;

#line 183
    df_0 _S1451 = _S428;

#line 183
    df_0 _S1452 = df_mul_0(&_S1450, &_S1451);

#line 183
    df_0 _S1453 = _S1449;

#line 183
    df_0 _S1454 = _S1452;

#line 183
    df_0 _S1455 = df_add_0(&_S1453, &_S1454);

#line 183
    df_0 _S1456 = _S428;

#line 183
    df_0 _S1457 = _S481;

#line 183
    df_0 _S1458 = df_mul_0(&_S1456, &_S1457);

#line 183
    df_0 _S1459 = _S481;

#line 183
    df_0 _S1460 = _S428;

#line 183
    df_0 _S1461 = df_mul_0(&_S1459, &_S1460);

#line 183
    df_0 _S1462 = _S1458;

#line 183
    df_0 _S1463 = _S1461;

#line 183
    df_0 _S1464 = df_add_0(&_S1462, &_S1463);

#line 183
    df_0 _S1465 = _S428;

#line 183
    df_0 _S1466 = _S258;

#line 183
    df_0 _S1467 = df_mul_0(&_S1465, &_S1466);

#line 183
    df_0 _S1468 = _S258;

#line 183
    df_0 _S1469 = _S428;

#line 183
    df_0 _S1470 = df_mul_0(&_S1468, &_S1469);

#line 183
    df_0 _S1471 = _S1467;

#line 183
    df_0 _S1472 = _S1470;

#line 183
    df_0 _S1473 = df_add_0(&_S1471, &_S1472);

#line 183
    df_0 _S1474 = _S428;

#line 183
    df_0 _S1475 = _S261;

#line 183
    df_0 _S1476 = df_mul_0(&_S1474, &_S1475);

#line 183
    df_0 _S1477 = _S261;

#line 183
    df_0 _S1478 = _S428;

#line 183
    df_0 _S1479 = df_mul_0(&_S1477, &_S1478);

#line 183
    df_0 _S1480 = _S1476;

#line 183
    df_0 _S1481 = _S1479;

#line 183
    df_0 _S1482 = df_add_0(&_S1480, &_S1481);

#line 183
    df_0 _S1483 = _S428;

#line 183
    df_0 _S1484 = _S464;

#line 183
    df_0 _S1485 = df_mul_0(&_S1483, &_S1484);

#line 183
    df_0 _S1486 = _S464;

#line 183
    df_0 _S1487 = _S428;

#line 183
    df_0 _S1488 = df_mul_0(&_S1486, &_S1487);

#line 183
    df_0 _S1489 = _S1485;

#line 183
    df_0 _S1490 = _S1488;

#line 183
    df_0 _S1491 = df_add_0(&_S1489, &_S1490);

#line 183
    df_0 _S1492 = _S428;

#line 183
    df_0 _S1493 = _S466;

#line 183
    df_0 _S1494 = df_mul_0(&_S1492, &_S1493);

#line 183
    df_0 _S1495 = _S466;

#line 183
    df_0 _S1496 = _S428;

#line 183
    df_0 _S1497 = df_mul_0(&_S1495, &_S1496);

#line 183
    df_0 _S1498 = _S1494;

#line 183
    df_0 _S1499 = _S1497;

#line 183
    df_0 _S1500 = df_add_0(&_S1498, &_S1499);

#line 183
    df_0 _S1501 = _S475;

#line 183
    df_0 _S1502 = _S475;

#line 183
    df_0 _S1503 = df_mul_0(&_S1501, &_S1502);

#line 183
    df_0 _S1504 = _S1503;

#line 183
    df_0 _S1505 = _S1503;

#line 183
    df_0 _S1506 = df_add_0(&_S1504, &_S1505);

#line 183
    df_0 _S1507 = _S449;

#line 183
    df_0 _S1508 = _S469;

#line 183
    df_0 _S1509 = df_mul_0(&_S1507, &_S1508);

#line 183
    df_0 _S1510 = _S475;

#line 183
    df_0 _S1511 = _S478;

#line 183
    df_0 _S1512 = df_mul_0(&_S1510, &_S1511);

#line 183
    df_0 _S1513 = _S478;

#line 183
    df_0 _S1514 = _S475;

#line 183
    df_0 _S1515 = df_mul_0(&_S1513, &_S1514);

#line 183
    df_0 _S1516 = _S469;

#line 183
    df_0 _S1517 = _S449;

#line 183
    df_0 _S1518 = df_mul_0(&_S1516, &_S1517);

#line 183
    df_0 _S1519 = _S1509;

#line 183
    df_0 _S1520 = _S1512;

#line 183
    df_0 _S1521 = df_add_0(&_S1519, &_S1520);

#line 183
    df_0 _S1522 = _S1521;

#line 183
    df_0 _S1523 = _S1515;

#line 183
    df_0 _S1524 = df_add_0(&_S1522, &_S1523);

#line 183
    df_0 _S1525 = _S1524;

#line 183
    df_0 _S1526 = _S1518;

#line 183
    df_0 _S1527 = df_add_0(&_S1525, &_S1526);

#line 183
    df_0 _S1528 = _S475;

#line 183
    df_0 _S1529 = _S437;

#line 183
    df_0 _S1530 = df_mul_0(&_S1528, &_S1529);

#line 183
    df_0 _S1531 = _S437;

#line 183
    df_0 _S1532 = _S475;

#line 183
    df_0 _S1533 = df_mul_0(&_S1531, &_S1532);

#line 183
    df_0 _S1534 = _S1530;

#line 183
    df_0 _S1535 = _S1533;

#line 183
    df_0 _S1536 = df_add_0(&_S1534, &_S1535);

#line 183
    df_0 _S1537 = _S475;

#line 183
    df_0 _S1538 = _S481;

#line 183
    df_0 _S1539 = df_mul_0(&_S1537, &_S1538);

#line 183
    df_0 _S1540 = _S481;

#line 183
    df_0 _S1541 = _S475;

#line 183
    df_0 _S1542 = df_mul_0(&_S1540, &_S1541);

#line 183
    df_0 _S1543 = _S1539;

#line 183
    df_0 _S1544 = _S1542;

#line 183
    df_0 _S1545 = df_add_0(&_S1543, &_S1544);

#line 183
    df_0 _S1546 = _S240;

#line 183
    df_0 _S1547 = _S469;

#line 183
    df_0 _S1548 = df_mul_0(&_S1546, &_S1547);

#line 183
    df_0 _S1549 = _S475;

#line 183
    df_0 _S1550 = _S258;

#line 183
    df_0 _S1551 = df_mul_0(&_S1549, &_S1550);

#line 183
    df_0 _S1552 = _S258;

#line 183
    df_0 _S1553 = _S475;

#line 183
    df_0 _S1554 = df_mul_0(&_S1552, &_S1553);

#line 183
    df_0 _S1555 = _S469;

#line 183
    df_0 _S1556 = _S240;

#line 183
    df_0 _S1557 = df_mul_0(&_S1555, &_S1556);

#line 183
    df_0 _S1558 = _S1548;

#line 183
    df_0 _S1559 = _S1551;

#line 183
    df_0 _S1560 = df_add_0(&_S1558, &_S1559);

#line 183
    df_0 _S1561 = _S1560;

#line 183
    df_0 _S1562 = _S1554;

#line 183
    df_0 _S1563 = df_add_0(&_S1561, &_S1562);

#line 183
    df_0 _S1564 = _S1563;

#line 183
    df_0 _S1565 = _S1557;

#line 183
    df_0 _S1566 = df_add_0(&_S1564, &_S1565);

#line 183
    df_0 _S1567 = _S475;

#line 183
    df_0 _S1568 = c1_0;

#line 183
    df_0 _S1569 = df_mul_0(&_S1567, &_S1568);

#line 183
    df_0 _S1570 = c1_0;

#line 183
    df_0 _S1571 = _S475;

#line 183
    df_0 _S1572 = df_mul_0(&_S1570, &_S1571);

#line 183
    df_0 _S1573 = _S1569;

#line 183
    df_0 _S1574 = _S1572;

#line 183
    df_0 _S1575 = df_add_0(&_S1573, &_S1574);

#line 183
    df_0 _S1576 = _S475;

#line 183
    df_0 _S1577 = _S261;

#line 183
    df_0 _S1578 = df_mul_0(&_S1576, &_S1577);

#line 183
    df_0 _S1579 = _S261;

#line 183
    df_0 _S1580 = _S475;

#line 183
    df_0 _S1581 = df_mul_0(&_S1579, &_S1580);

#line 183
    df_0 _S1582 = _S1578;

#line 183
    df_0 _S1583 = _S1581;

#line 183
    df_0 _S1584 = df_add_0(&_S1582, &_S1583);

#line 183
    df_0 _S1585 = _S351;

#line 183
    df_0 _S1586 = _S469;

#line 183
    df_0 _S1587 = df_mul_0(&_S1585, &_S1586);

#line 183
    df_0 _S1588 = _S475;

#line 183
    df_0 _S1589 = _S464;

#line 183
    df_0 _S1590 = df_mul_0(&_S1588, &_S1589);

#line 183
    df_0 _S1591 = _S464;

#line 183
    df_0 _S1592 = _S475;

#line 183
    df_0 _S1593 = df_mul_0(&_S1591, &_S1592);

#line 183
    df_0 _S1594 = _S469;

#line 183
    df_0 _S1595 = _S351;

#line 183
    df_0 _S1596 = df_mul_0(&_S1594, &_S1595);

#line 183
    df_0 _S1597 = _S1587;

#line 183
    df_0 _S1598 = _S1590;

#line 183
    df_0 _S1599 = df_add_0(&_S1597, &_S1598);

#line 183
    df_0 _S1600 = _S1599;

#line 183
    df_0 _S1601 = _S1593;

#line 183
    df_0 _S1602 = df_add_0(&_S1600, &_S1601);

#line 183
    df_0 _S1603 = _S1602;

#line 183
    df_0 _S1604 = _S1596;

#line 183
    df_0 _S1605 = df_add_0(&_S1603, &_S1604);

#line 183
    df_0 _S1606 = _S475;

#line 183
    df_0 _S1607 = _S296;

#line 183
    df_0 _S1608 = df_mul_0(&_S1606, &_S1607);

#line 183
    df_0 _S1609 = _S296;

#line 183
    df_0 _S1610 = _S475;

#line 183
    df_0 _S1611 = df_mul_0(&_S1609, &_S1610);

#line 183
    df_0 _S1612 = _S1608;

#line 183
    df_0 _S1613 = _S1611;

#line 183
    df_0 _S1614 = df_add_0(&_S1612, &_S1613);

#line 183
    df_0 _S1615 = _S475;

#line 183
    df_0 _S1616 = _S466;

#line 183
    df_0 _S1617 = df_mul_0(&_S1615, &_S1616);

#line 183
    df_0 _S1618 = _S466;

#line 183
    df_0 _S1619 = _S475;

#line 183
    df_0 _S1620 = df_mul_0(&_S1618, &_S1619);

#line 183
    df_0 _S1621 = _S1617;

#line 183
    df_0 _S1622 = _S1620;

#line 183
    df_0 _S1623 = df_add_0(&_S1621, &_S1622);

#line 183
    df_0 _S1624 = _S478;

#line 183
    df_0 _S1625 = _S478;

#line 183
    df_0 _S1626 = df_mul_0(&_S1624, &_S1625);

#line 183
    df_0 _S1627 = _S1626;

#line 183
    df_0 _S1628 = _S1626;

#line 183
    df_0 _S1629 = df_add_0(&_S1627, &_S1628);

#line 183
    df_0 _S1630 = _S478;

#line 183
    df_0 _S1631 = _S437;

#line 183
    df_0 _S1632 = df_mul_0(&_S1630, &_S1631);

#line 183
    df_0 _S1633 = _S437;

#line 183
    df_0 _S1634 = _S478;

#line 183
    df_0 _S1635 = df_mul_0(&_S1633, &_S1634);

#line 183
    df_0 _S1636 = _S1632;

#line 183
    df_0 _S1637 = _S1635;

#line 183
    df_0 _S1638 = df_add_0(&_S1636, &_S1637);

#line 183
    df_0 _S1639 = _S478;

#line 183
    df_0 _S1640 = _S481;

#line 183
    df_0 _S1641 = df_mul_0(&_S1639, &_S1640);

#line 183
    df_0 _S1642 = _S481;

#line 183
    df_0 _S1643 = _S478;

#line 183
    df_0 _S1644 = df_mul_0(&_S1642, &_S1643);

#line 183
    df_0 _S1645 = _S1641;

#line 183
    df_0 _S1646 = _S1644;

#line 183
    df_0 _S1647 = df_add_0(&_S1645, &_S1646);

#line 183
    df_0 _S1648 = _S478;

#line 183
    df_0 _S1649 = _S258;

#line 183
    df_0 _S1650 = df_mul_0(&_S1648, &_S1649);

#line 183
    df_0 _S1651 = _S258;

#line 183
    df_0 _S1652 = _S478;

#line 183
    df_0 _S1653 = df_mul_0(&_S1651, &_S1652);

#line 183
    df_0 _S1654 = _S1650;

#line 183
    df_0 _S1655 = _S1653;

#line 183
    df_0 _S1656 = df_add_0(&_S1654, &_S1655);

#line 183
    df_0 _S1657 = _S478;

#line 183
    df_0 _S1658 = c1_0;

#line 183
    df_0 _S1659 = df_mul_0(&_S1657, &_S1658);

#line 183
    df_0 _S1660 = c1_0;

#line 183
    df_0 _S1661 = _S478;

#line 183
    df_0 _S1662 = df_mul_0(&_S1660, &_S1661);

#line 183
    df_0 _S1663 = _S1659;

#line 183
    df_0 _S1664 = _S1662;

#line 183
    df_0 _S1665 = df_add_0(&_S1663, &_S1664);

#line 183
    df_0 _S1666 = _S478;

#line 183
    df_0 _S1667 = _S261;

#line 183
    df_0 _S1668 = df_mul_0(&_S1666, &_S1667);

#line 183
    df_0 _S1669 = _S261;

#line 183
    df_0 _S1670 = _S478;

#line 183
    df_0 _S1671 = df_mul_0(&_S1669, &_S1670);

#line 183
    df_0 _S1672 = _S1548;

#line 183
    df_0 _S1673 = _S1668;

#line 183
    df_0 _S1674 = df_add_0(&_S1672, &_S1673);

#line 183
    df_0 _S1675 = _S1674;

#line 183
    df_0 _S1676 = _S1671;

#line 183
    df_0 _S1677 = df_add_0(&_S1675, &_S1676);

#line 183
    df_0 _S1678 = _S1677;

#line 183
    df_0 _S1679 = _S1557;

#line 183
    df_0 _S1680 = df_add_0(&_S1678, &_S1679);

#line 183
    df_0 _S1681 = _S478;

#line 183
    df_0 _S1682 = _S464;

#line 183
    df_0 _S1683 = df_mul_0(&_S1681, &_S1682);

#line 183
    df_0 _S1684 = _S464;

#line 183
    df_0 _S1685 = _S478;

#line 183
    df_0 _S1686 = df_mul_0(&_S1684, &_S1685);

#line 183
    df_0 _S1687 = _S1683;

#line 183
    df_0 _S1688 = _S1686;

#line 183
    df_0 _S1689 = df_add_0(&_S1687, &_S1688);

#line 183
    df_0 _S1690 = _S478;

#line 183
    df_0 _S1691 = _S296;

#line 183
    df_0 _S1692 = df_mul_0(&_S1690, &_S1691);

#line 183
    df_0 _S1693 = _S296;

#line 183
    df_0 _S1694 = _S478;

#line 183
    df_0 _S1695 = df_mul_0(&_S1693, &_S1694);

#line 183
    df_0 _S1696 = _S1692;

#line 183
    df_0 _S1697 = _S1695;

#line 183
    df_0 _S1698 = df_add_0(&_S1696, &_S1697);

#line 183
    df_0 _S1699 = _S478;

#line 183
    df_0 _S1700 = _S466;

#line 183
    df_0 _S1701 = df_mul_0(&_S1699, &_S1700);

#line 183
    df_0 _S1702 = _S466;

#line 183
    df_0 _S1703 = _S478;

#line 183
    df_0 _S1704 = df_mul_0(&_S1702, &_S1703);

#line 183
    df_0 _S1705 = _S1587;

#line 183
    df_0 _S1706 = _S1701;

#line 183
    df_0 _S1707 = df_add_0(&_S1705, &_S1706);

#line 183
    df_0 _S1708 = _S1707;

#line 183
    df_0 _S1709 = _S1704;

#line 183
    df_0 _S1710 = df_add_0(&_S1708, &_S1709);

#line 183
    df_0 _S1711 = _S1710;

#line 183
    df_0 _S1712 = _S1596;

#line 183
    df_0 _S1713 = df_add_0(&_S1711, &_S1712);

#line 183
    df_0 _S1714 = _S437;

#line 183
    df_0 _S1715 = _S481;

#line 183
    df_0 _S1716 = df_mul_0(&_S1714, &_S1715);

#line 183
    df_0 _S1717 = _S481;

#line 183
    df_0 _S1718 = _S437;

#line 183
    df_0 _S1719 = df_mul_0(&_S1717, &_S1718);

#line 183
    df_0 _S1720 = _S1716;

#line 183
    df_0 _S1721 = _S1719;

#line 183
    df_0 _S1722 = df_add_0(&_S1720, &_S1721);

#line 183
    df_0 _S1723 = _S437;

#line 183
    df_0 _S1724 = _S258;

#line 183
    df_0 _S1725 = df_mul_0(&_S1723, &_S1724);

#line 183
    df_0 _S1726 = _S258;

#line 183
    df_0 _S1727 = _S437;

#line 183
    df_0 _S1728 = df_mul_0(&_S1726, &_S1727);

#line 183
    df_0 _S1729 = _S1725;

#line 183
    df_0 _S1730 = _S1728;

#line 183
    df_0 _S1731 = df_add_0(&_S1729, &_S1730);

#line 183
    df_0 _S1732 = _S437;

#line 183
    df_0 _S1733 = _S261;

#line 183
    df_0 _S1734 = df_mul_0(&_S1732, &_S1733);

#line 183
    df_0 _S1735 = _S261;

#line 183
    df_0 _S1736 = _S437;

#line 183
    df_0 _S1737 = df_mul_0(&_S1735, &_S1736);

#line 183
    df_0 _S1738 = _S1734;

#line 183
    df_0 _S1739 = _S1737;

#line 183
    df_0 _S1740 = df_add_0(&_S1738, &_S1739);

#line 183
    df_0 _S1741 = _S437;

#line 183
    df_0 _S1742 = _S464;

#line 183
    df_0 _S1743 = df_mul_0(&_S1741, &_S1742);

#line 183
    df_0 _S1744 = _S464;

#line 183
    df_0 _S1745 = _S437;

#line 183
    df_0 _S1746 = df_mul_0(&_S1744, &_S1745);

#line 183
    df_0 _S1747 = _S1743;

#line 183
    df_0 _S1748 = _S1746;

#line 183
    df_0 _S1749 = df_add_0(&_S1747, &_S1748);

#line 183
    df_0 _S1750 = _S437;

#line 183
    df_0 _S1751 = _S466;

#line 183
    df_0 _S1752 = df_mul_0(&_S1750, &_S1751);

#line 183
    df_0 _S1753 = _S466;

#line 183
    df_0 _S1754 = _S437;

#line 183
    df_0 _S1755 = df_mul_0(&_S1753, &_S1754);

#line 183
    df_0 _S1756 = _S1752;

#line 183
    df_0 _S1757 = _S1755;

#line 183
    df_0 _S1758 = df_add_0(&_S1756, &_S1757);

#line 183
    df_0 _S1759 = _S481;

#line 183
    df_0 _S1760 = _S481;

#line 183
    df_0 _S1761 = df_mul_0(&_S1759, &_S1760);

#line 183
    df_0 _S1762 = _S1761;

#line 183
    df_0 _S1763 = _S1761;

#line 183
    df_0 _S1764 = df_add_0(&_S1762, &_S1763);

#line 183
    df_0 _S1765 = _S481;

#line 183
    df_0 _S1766 = _S258;

#line 183
    df_0 _S1767 = df_mul_0(&_S1765, &_S1766);

#line 183
    df_0 _S1768 = _S258;

#line 183
    df_0 _S1769 = _S481;

#line 183
    df_0 _S1770 = df_mul_0(&_S1768, &_S1769);

#line 183
    df_0 _S1771 = _S1380;

#line 183
    df_0 _S1772 = _S1767;

#line 183
    df_0 _S1773 = df_add_0(&_S1771, &_S1772);

#line 183
    df_0 _S1774 = _S1773;

#line 183
    df_0 _S1775 = _S1770;

#line 183
    df_0 _S1776 = df_add_0(&_S1774, &_S1775);

#line 183
    df_0 _S1777 = _S1776;

#line 183
    df_0 _S1778 = _S1389;

#line 183
    df_0 _S1779 = df_add_0(&_S1777, &_S1778);

#line 183
    df_0 _S1780 = _S481;

#line 183
    df_0 _S1781 = c1_0;

#line 183
    df_0 _S1782 = df_mul_0(&_S1780, &_S1781);

#line 183
    df_0 _S1783 = c1_0;

#line 183
    df_0 _S1784 = _S481;

#line 183
    df_0 _S1785 = df_mul_0(&_S1783, &_S1784);

#line 183
    df_0 _S1786 = _S1782;

#line 183
    df_0 _S1787 = _S1785;

#line 183
    df_0 _S1788 = df_add_0(&_S1786, &_S1787);

#line 183
    df_0 _S1789 = _S481;

#line 183
    df_0 _S1790 = _S261;

#line 183
    df_0 _S1791 = df_mul_0(&_S1789, &_S1790);

#line 183
    df_0 _S1792 = _S261;

#line 183
    df_0 _S1793 = _S481;

#line 183
    df_0 _S1794 = df_mul_0(&_S1792, &_S1793);

#line 183
    df_0 _S1795 = _S1791;

#line 183
    df_0 _S1796 = _S1794;

#line 183
    df_0 _S1797 = df_add_0(&_S1795, &_S1796);

#line 183
    df_0 _S1798 = _S481;

#line 183
    df_0 _S1799 = _S464;

#line 183
    df_0 _S1800 = df_mul_0(&_S1798, &_S1799);

#line 183
    df_0 _S1801 = _S464;

#line 183
    df_0 _S1802 = _S481;

#line 183
    df_0 _S1803 = df_mul_0(&_S1801, &_S1802);

#line 183
    df_0 _S1804 = _S1419;

#line 183
    df_0 _S1805 = _S1800;

#line 183
    df_0 _S1806 = df_add_0(&_S1804, &_S1805);

#line 183
    df_0 _S1807 = _S1806;

#line 183
    df_0 _S1808 = _S1803;

#line 183
    df_0 _S1809 = df_add_0(&_S1807, &_S1808);

#line 183
    df_0 _S1810 = _S1809;

#line 183
    df_0 _S1811 = _S1428;

#line 183
    df_0 _S1812 = df_add_0(&_S1810, &_S1811);

#line 183
    df_0 _S1813 = _S481;

#line 183
    df_0 _S1814 = _S296;

#line 183
    df_0 _S1815 = df_mul_0(&_S1813, &_S1814);

#line 183
    df_0 _S1816 = _S296;

#line 183
    df_0 _S1817 = _S481;

#line 183
    df_0 _S1818 = df_mul_0(&_S1816, &_S1817);

#line 183
    df_0 _S1819 = _S1815;

#line 183
    df_0 _S1820 = _S1818;

#line 183
    df_0 _S1821 = df_add_0(&_S1819, &_S1820);

#line 183
    df_0 _S1822 = _S481;

#line 183
    df_0 _S1823 = _S466;

#line 183
    df_0 _S1824 = df_mul_0(&_S1822, &_S1823);

#line 183
    df_0 _S1825 = _S466;

#line 183
    df_0 _S1826 = _S481;

#line 183
    df_0 _S1827 = df_mul_0(&_S1825, &_S1826);

#line 183
    df_0 _S1828 = _S1824;

#line 183
    df_0 _S1829 = _S1827;

#line 183
    df_0 _S1830 = df_add_0(&_S1828, &_S1829);

#line 183
    df_0 _S1831 = _S258;

#line 183
    df_0 _S1832 = _S258;

#line 183
    df_0 _S1833 = df_mul_0(&_S1831, &_S1832);

#line 183
    df_0 _S1834 = _S1833;

#line 183
    df_0 _S1835 = _S1833;

#line 183
    df_0 _S1836 = df_add_0(&_S1834, &_S1835);

#line 183
    df_0 _S1837 = _S258;

#line 183
    df_0 _S1838 = c1_0;

#line 183
    df_0 _S1839 = df_mul_0(&_S1837, &_S1838);

#line 183
    df_0 _S1840 = c1_0;

#line 183
    df_0 _S1841 = _S258;

#line 183
    df_0 _S1842 = df_mul_0(&_S1840, &_S1841);

#line 183
    df_0 _S1843 = _S1839;

#line 183
    df_0 _S1844 = _S1842;

#line 183
    df_0 _S1845 = df_add_0(&_S1843, &_S1844);

#line 183
    df_0 _S1846 = _S258;

#line 183
    df_0 _S1847 = _S261;

#line 183
    df_0 _S1848 = df_mul_0(&_S1846, &_S1847);

#line 183
    df_0 _S1849 = _S261;

#line 183
    df_0 _S1850 = _S258;

#line 183
    df_0 _S1851 = df_mul_0(&_S1849, &_S1850);

#line 183
    df_0 _S1852 = _S1848;

#line 183
    df_0 _S1853 = _S1851;

#line 183
    df_0 _S1854 = df_add_0(&_S1852, &_S1853);

#line 183
    df_0 _S1855 = _S258;

#line 183
    df_0 _S1856 = _S464;

#line 183
    df_0 _S1857 = df_mul_0(&_S1855, &_S1856);

#line 183
    df_0 _S1858 = _S464;

#line 183
    df_0 _S1859 = _S258;

#line 183
    df_0 _S1860 = df_mul_0(&_S1858, &_S1859);

#line 183
    df_0 _S1861 = _S1857;

#line 183
    df_0 _S1862 = _S1860;

#line 183
    df_0 _S1863 = df_add_0(&_S1861, &_S1862);

#line 183
    df_0 _S1864 = _S258;

#line 183
    df_0 _S1865 = _S296;

#line 183
    df_0 _S1866 = df_mul_0(&_S1864, &_S1865);

#line 183
    df_0 _S1867 = _S296;

#line 183
    df_0 _S1868 = _S258;

#line 183
    df_0 _S1869 = df_mul_0(&_S1867, &_S1868);

#line 183
    df_0 _S1870 = _S1866;

#line 183
    df_0 _S1871 = _S1869;

#line 183
    df_0 _S1872 = df_add_0(&_S1870, &_S1871);

#line 183
    df_0 _S1873 = _S258;

#line 183
    df_0 _S1874 = _S466;

#line 183
    df_0 _S1875 = df_mul_0(&_S1873, &_S1874);

#line 183
    df_0 _S1876 = _S466;

#line 183
    df_0 _S1877 = _S258;

#line 183
    df_0 _S1878 = df_mul_0(&_S1876, &_S1877);

#line 183
    df_0 _S1879 = _S1875;

#line 183
    df_0 _S1880 = _S1878;

#line 183
    df_0 _S1881 = df_add_0(&_S1879, &_S1880);

#line 183
    df_0 _S1882 = c1_0;

#line 183
    df_0 _S1883 = _S261;

#line 183
    df_0 _S1884 = df_mul_0(&_S1882, &_S1883);

#line 183
    df_0 _S1885 = _S261;

#line 183
    df_0 _S1886 = c1_0;

#line 183
    df_0 _S1887 = df_mul_0(&_S1885, &_S1886);

#line 183
    df_0 _S1888 = _S1884;

#line 183
    df_0 _S1889 = _S1887;

#line 183
    df_0 _S1890 = df_add_0(&_S1888, &_S1889);

#line 183
    df_0 _S1891 = c1_0;

#line 183
    df_0 _S1892 = _S464;

#line 183
    df_0 _S1893 = df_mul_0(&_S1891, &_S1892);

#line 183
    df_0 _S1894 = _S464;

#line 183
    df_0 _S1895 = c1_0;

#line 183
    df_0 _S1896 = df_mul_0(&_S1894, &_S1895);

#line 183
    df_0 _S1897 = _S1893;

#line 183
    df_0 _S1898 = _S1896;

#line 183
    df_0 _S1899 = df_add_0(&_S1897, &_S1898);

#line 183
    df_0 _S1900 = c1_0;

#line 183
    df_0 _S1901 = _S466;

#line 183
    df_0 _S1902 = df_mul_0(&_S1900, &_S1901);

#line 183
    df_0 _S1903 = _S466;

#line 183
    df_0 _S1904 = c1_0;

#line 183
    df_0 _S1905 = df_mul_0(&_S1903, &_S1904);

#line 183
    df_0 _S1906 = _S1902;

#line 183
    df_0 _S1907 = _S1905;

#line 183
    df_0 _S1908 = df_add_0(&_S1906, &_S1907);

#line 183
    df_0 _S1909 = _S261;

#line 183
    df_0 _S1910 = _S261;

#line 183
    df_0 _S1911 = df_mul_0(&_S1909, &_S1910);

#line 183
    df_0 _S1912 = _S1911;

#line 183
    df_0 _S1913 = _S1911;

#line 183
    df_0 _S1914 = df_add_0(&_S1912, &_S1913);

#line 183
    df_0 _S1915 = _S261;

#line 183
    df_0 _S1916 = _S464;

#line 183
    df_0 _S1917 = df_mul_0(&_S1915, &_S1916);

#line 183
    df_0 _S1918 = _S464;

#line 183
    df_0 _S1919 = _S261;

#line 183
    df_0 _S1920 = df_mul_0(&_S1918, &_S1919);

#line 183
    df_0 _S1921 = _S1917;

#line 183
    df_0 _S1922 = _S1920;

#line 183
    df_0 _S1923 = df_add_0(&_S1921, &_S1922);

#line 183
    df_0 _S1924 = _S261;

#line 183
    df_0 _S1925 = _S296;

#line 183
    df_0 _S1926 = df_mul_0(&_S1924, &_S1925);

#line 183
    df_0 _S1927 = _S296;

#line 183
    df_0 _S1928 = _S261;

#line 183
    df_0 _S1929 = df_mul_0(&_S1927, &_S1928);

#line 183
    df_0 _S1930 = _S1926;

#line 183
    df_0 _S1931 = _S1929;

#line 183
    df_0 _S1932 = df_add_0(&_S1930, &_S1931);

#line 183
    df_0 _S1933 = _S261;

#line 183
    df_0 _S1934 = _S466;

#line 183
    df_0 _S1935 = df_mul_0(&_S1933, &_S1934);

#line 183
    df_0 _S1936 = _S466;

#line 183
    df_0 _S1937 = _S261;

#line 183
    df_0 _S1938 = df_mul_0(&_S1936, &_S1937);

#line 183
    df_0 _S1939 = _S1935;

#line 183
    df_0 _S1940 = _S1938;

#line 183
    df_0 _S1941 = df_add_0(&_S1939, &_S1940);

#line 183
    df_0 _S1942 = _S464;

#line 183
    df_0 _S1943 = _S464;

#line 183
    df_0 _S1944 = df_mul_0(&_S1942, &_S1943);

#line 183
    df_0 _S1945 = _S1944;

#line 183
    df_0 _S1946 = _S1944;

#line 183
    df_0 _S1947 = df_add_0(&_S1945, &_S1946);

#line 183
    df_0 _S1948 = _S464;

#line 183
    df_0 _S1949 = _S296;

#line 183
    df_0 _S1950 = df_mul_0(&_S1948, &_S1949);

#line 183
    df_0 _S1951 = _S296;

#line 183
    df_0 _S1952 = _S464;

#line 183
    df_0 _S1953 = df_mul_0(&_S1951, &_S1952);

#line 183
    df_0 _S1954 = _S1950;

#line 183
    df_0 _S1955 = _S1953;

#line 183
    df_0 _S1956 = df_add_0(&_S1954, &_S1955);

#line 183
    df_0 _S1957 = _S464;

#line 183
    df_0 _S1958 = _S466;

#line 183
    df_0 _S1959 = df_mul_0(&_S1957, &_S1958);

#line 183
    df_0 _S1960 = _S466;

#line 183
    df_0 _S1961 = _S464;

#line 183
    df_0 _S1962 = df_mul_0(&_S1960, &_S1961);

#line 183
    df_0 _S1963 = _S1959;

#line 183
    df_0 _S1964 = _S1962;

#line 183
    df_0 _S1965 = df_add_0(&_S1963, &_S1964);

#line 183
    df_0 _S1966 = _S296;

#line 183
    df_0 _S1967 = _S466;

#line 183
    df_0 _S1968 = df_mul_0(&_S1966, &_S1967);

#line 183
    df_0 _S1969 = _S466;

#line 183
    df_0 _S1970 = _S296;

#line 183
    df_0 _S1971 = df_mul_0(&_S1969, &_S1970);

#line 183
    df_0 _S1972 = _S1968;

#line 183
    df_0 _S1973 = _S1971;

#line 183
    df_0 _S1974 = df_add_0(&_S1972, &_S1973);

#line 183
    df_0 _S1975 = _S466;

#line 183
    df_0 _S1976 = _S466;

#line 183
    df_0 _S1977 = df_mul_0(&_S1975, &_S1976);

#line 183
    df_0 _S1978 = _S1977;

#line 183
    df_0 _S1979 = _S1977;

#line 183
    df_0 _S1980 = df_add_0(&_S1978, &_S1979);

#line 183
    df_0 _S1981 = _S501;

#line 183
    df_0 _S1982 = _S501;

#line 183
    df_0 _S1983 = df_mul_0(&_S1981, &_S1982);

#line 183
    df_0 _S1984 = _S1983;

#line 183
    df_0 _S1985 = _S1983;

#line 183
    df_0 _S1986 = df_add_0(&_S1984, &_S1985);

#line 183
    df_0 _S1987 = _S446;

#line 183
    df_0 _S1988 = _S498;

#line 183
    df_0 _S1989 = df_mul_0(&_S1987, &_S1988);

#line 183
    df_0 _S1990 = _S501;

#line 183
    df_0 _S1991 = _S504;

#line 183
    df_0 _S1992 = df_mul_0(&_S1990, &_S1991);

#line 183
    df_0 _S1993 = _S504;

#line 183
    df_0 _S1994 = _S501;

#line 183
    df_0 _S1995 = df_mul_0(&_S1993, &_S1994);

#line 183
    df_0 _S1996 = _S498;

#line 183
    df_0 _S1997 = _S446;

#line 183
    df_0 _S1998 = df_mul_0(&_S1996, &_S1997);

#line 183
    df_0 _S1999 = _S1989;

#line 183
    df_0 _S2000 = _S1992;

#line 183
    df_0 _S2001 = df_add_0(&_S1999, &_S2000);

#line 183
    df_0 _S2002 = _S2001;

#line 183
    df_0 _S2003 = _S1995;

#line 183
    df_0 _S2004 = df_add_0(&_S2002, &_S2003);

#line 183
    df_0 _S2005 = _S2004;

#line 183
    df_0 _S2006 = _S1998;

#line 183
    df_0 _S2007 = df_add_0(&_S2005, &_S2006);

#line 183
    df_0 _S2008 = _S501;

#line 183
    df_0 _S2009 = _S428;

#line 183
    df_0 _S2010 = df_mul_0(&_S2008, &_S2009);

#line 183
    df_0 _S2011 = _S428;

#line 183
    df_0 _S2012 = _S501;

#line 183
    df_0 _S2013 = df_mul_0(&_S2011, &_S2012);

#line 183
    df_0 _S2014 = _S2010;

#line 183
    df_0 _S2015 = _S2013;

#line 183
    df_0 _S2016 = df_add_0(&_S2014, &_S2015);

#line 183
    df_0 _S2017 = _S501;

#line 183
    df_0 _S2018 = _S507;

#line 183
    df_0 _S2019 = df_mul_0(&_S2017, &_S2018);

#line 183
    df_0 _S2020 = _S507;

#line 183
    df_0 _S2021 = _S501;

#line 183
    df_0 _S2022 = df_mul_0(&_S2020, &_S2021);

#line 183
    df_0 _S2023 = _S2019;

#line 183
    df_0 _S2024 = _S2022;

#line 183
    df_0 _S2025 = df_add_0(&_S2023, &_S2024);

#line 183
    df_0 _S2026 = _S449;

#line 183
    df_0 _S2027 = _S498;

#line 183
    df_0 _S2028 = df_mul_0(&_S2026, &_S2027);

#line 183
    df_0 _S2029 = _S501;

#line 183
    df_0 _S2030 = _S510;

#line 183
    df_0 _S2031 = df_mul_0(&_S2029, &_S2030);

#line 183
    df_0 _S2032 = _S510;

#line 183
    df_0 _S2033 = _S501;

#line 183
    df_0 _S2034 = df_mul_0(&_S2032, &_S2033);

#line 183
    df_0 _S2035 = _S498;

#line 183
    df_0 _S2036 = _S449;

#line 183
    df_0 _S2037 = df_mul_0(&_S2035, &_S2036);

#line 183
    df_0 _S2038 = _S2028;

#line 183
    df_0 _S2039 = _S2031;

#line 183
    df_0 _S2040 = df_add_0(&_S2038, &_S2039);

#line 183
    df_0 _S2041 = _S2040;

#line 183
    df_0 _S2042 = _S2034;

#line 183
    df_0 _S2043 = df_add_0(&_S2041, &_S2042);

#line 183
    df_0 _S2044 = _S2043;

#line 183
    df_0 _S2045 = _S2037;

#line 183
    df_0 _S2046 = df_add_0(&_S2044, &_S2045);

#line 183
    df_0 _S2047 = _S501;

#line 183
    df_0 _S2048 = _S437;

#line 183
    df_0 _S2049 = df_mul_0(&_S2047, &_S2048);

#line 183
    df_0 _S2050 = _S437;

#line 183
    df_0 _S2051 = _S501;

#line 183
    df_0 _S2052 = df_mul_0(&_S2050, &_S2051);

#line 183
    df_0 _S2053 = _S2049;

#line 183
    df_0 _S2054 = _S2052;

#line 183
    df_0 _S2055 = df_add_0(&_S2053, &_S2054);

#line 183
    df_0 _S2056 = _S501;

#line 183
    df_0 _S2057 = _S279;

#line 183
    df_0 _S2058 = df_mul_0(&_S2056, &_S2057);

#line 183
    df_0 _S2059 = _S279;

#line 183
    df_0 _S2060 = _S501;

#line 183
    df_0 _S2061 = df_mul_0(&_S2059, &_S2060);

#line 183
    df_0 _S2062 = _S2058;

#line 183
    df_0 _S2063 = _S2061;

#line 183
    df_0 _S2064 = df_add_0(&_S2062, &_S2063);

#line 183
    df_0 _S2065 = _S240;

#line 183
    df_0 _S2066 = _S498;

#line 183
    df_0 _S2067 = df_mul_0(&_S2065, &_S2066);

#line 183
    df_0 _S2068 = _S501;

#line 183
    df_0 _S2069 = _S282;

#line 183
    df_0 _S2070 = df_mul_0(&_S2068, &_S2069);

#line 183
    df_0 _S2071 = _S282;

#line 183
    df_0 _S2072 = _S501;

#line 183
    df_0 _S2073 = df_mul_0(&_S2071, &_S2072);

#line 183
    df_0 _S2074 = _S498;

#line 183
    df_0 _S2075 = _S240;

#line 183
    df_0 _S2076 = df_mul_0(&_S2074, &_S2075);

#line 183
    df_0 _S2077 = _S2067;

#line 183
    df_0 _S2078 = _S2070;

#line 183
    df_0 _S2079 = df_add_0(&_S2077, &_S2078);

#line 183
    df_0 _S2080 = _S2079;

#line 183
    df_0 _S2081 = _S2073;

#line 183
    df_0 _S2082 = df_add_0(&_S2080, &_S2081);

#line 183
    df_0 _S2083 = _S2082;

#line 183
    df_0 _S2084 = _S2076;

#line 183
    df_0 _S2085 = df_add_0(&_S2083, &_S2084);

#line 183
    df_0 _S2086 = _S501;

#line 183
    df_0 _S2087 = c1_0;

#line 183
    df_0 _S2088 = df_mul_0(&_S2086, &_S2087);

#line 183
    df_0 _S2089 = c1_0;

#line 183
    df_0 _S2090 = _S501;

#line 183
    df_0 _S2091 = df_mul_0(&_S2089, &_S2090);

#line 183
    df_0 _S2092 = _S2088;

#line 183
    df_0 _S2093 = _S2091;

#line 183
    df_0 _S2094 = df_add_0(&_S2092, &_S2093);

#line 183
    df_0 _S2095 = _S501;

#line 183
    df_0 _S2096 = _S493;

#line 183
    df_0 _S2097 = df_mul_0(&_S2095, &_S2096);

#line 183
    df_0 _S2098 = _S493;

#line 183
    df_0 _S2099 = _S501;

#line 183
    df_0 _S2100 = df_mul_0(&_S2098, &_S2099);

#line 183
    df_0 _S2101 = _S2097;

#line 183
    df_0 _S2102 = _S2100;

#line 183
    df_0 _S2103 = df_add_0(&_S2101, &_S2102);

#line 183
    df_0 _S2104 = _S351;

#line 183
    df_0 _S2105 = _S498;

#line 183
    df_0 _S2106 = df_mul_0(&_S2104, &_S2105);

#line 183
    df_0 _S2107 = _S501;

#line 183
    df_0 _S2108 = _S495;

#line 183
    df_0 _S2109 = df_mul_0(&_S2107, &_S2108);

#line 183
    df_0 _S2110 = _S495;

#line 183
    df_0 _S2111 = _S501;

#line 183
    df_0 _S2112 = df_mul_0(&_S2110, &_S2111);

#line 183
    df_0 _S2113 = _S498;

#line 183
    df_0 _S2114 = _S351;

#line 183
    df_0 _S2115 = df_mul_0(&_S2113, &_S2114);

#line 183
    df_0 _S2116 = _S2106;

#line 183
    df_0 _S2117 = _S2109;

#line 183
    df_0 _S2118 = df_add_0(&_S2116, &_S2117);

#line 183
    df_0 _S2119 = _S2118;

#line 183
    df_0 _S2120 = _S2112;

#line 183
    df_0 _S2121 = df_add_0(&_S2119, &_S2120);

#line 183
    df_0 _S2122 = _S2121;

#line 183
    df_0 _S2123 = _S2115;

#line 183
    df_0 _S2124 = df_add_0(&_S2122, &_S2123);

#line 183
    df_0 _S2125 = _S501;

#line 183
    df_0 _S2126 = _S296;

#line 183
    df_0 _S2127 = df_mul_0(&_S2125, &_S2126);

#line 183
    df_0 _S2128 = _S296;

#line 183
    df_0 _S2129 = _S501;

#line 183
    df_0 _S2130 = df_mul_0(&_S2128, &_S2129);

#line 183
    df_0 _S2131 = _S2127;

#line 183
    df_0 _S2132 = _S2130;

#line 183
    df_0 _S2133 = df_add_0(&_S2131, &_S2132);

#line 183
    df_0 _S2134 = _S504;

#line 183
    df_0 _S2135 = _S504;

#line 183
    df_0 _S2136 = df_mul_0(&_S2134, &_S2135);

#line 183
    df_0 _S2137 = _S2136;

#line 183
    df_0 _S2138 = _S2136;

#line 183
    df_0 _S2139 = df_add_0(&_S2137, &_S2138);

#line 183
    df_0 _S2140 = _S504;

#line 183
    df_0 _S2141 = _S428;

#line 183
    df_0 _S2142 = df_mul_0(&_S2140, &_S2141);

#line 183
    df_0 _S2143 = _S428;

#line 183
    df_0 _S2144 = _S504;

#line 183
    df_0 _S2145 = df_mul_0(&_S2143, &_S2144);

#line 183
    df_0 _S2146 = _S2142;

#line 183
    df_0 _S2147 = _S2145;

#line 183
    df_0 _S2148 = df_add_0(&_S2146, &_S2147);

#line 183
    df_0 _S2149 = _S452;

#line 183
    df_0 _S2150 = _S498;

#line 183
    df_0 _S2151 = df_mul_0(&_S2149, &_S2150);

#line 183
    df_0 _S2152 = _S504;

#line 183
    df_0 _S2153 = _S507;

#line 183
    df_0 _S2154 = df_mul_0(&_S2152, &_S2153);

#line 183
    df_0 _S2155 = _S507;

#line 183
    df_0 _S2156 = _S504;

#line 183
    df_0 _S2157 = df_mul_0(&_S2155, &_S2156);

#line 183
    df_0 _S2158 = _S498;

#line 183
    df_0 _S2159 = _S452;

#line 183
    df_0 _S2160 = df_mul_0(&_S2158, &_S2159);

#line 183
    df_0 _S2161 = _S2151;

#line 183
    df_0 _S2162 = _S2154;

#line 183
    df_0 _S2163 = df_add_0(&_S2161, &_S2162);

#line 183
    df_0 _S2164 = _S2163;

#line 183
    df_0 _S2165 = _S2157;

#line 183
    df_0 _S2166 = df_add_0(&_S2164, &_S2165);

#line 183
    df_0 _S2167 = _S2166;

#line 183
    df_0 _S2168 = _S2160;

#line 183
    df_0 _S2169 = df_add_0(&_S2167, &_S2168);

#line 183
    df_0 _S2170 = _S504;

#line 183
    df_0 _S2171 = _S510;

#line 183
    df_0 _S2172 = df_mul_0(&_S2170, &_S2171);

#line 183
    df_0 _S2173 = _S510;

#line 183
    df_0 _S2174 = _S504;

#line 183
    df_0 _S2175 = df_mul_0(&_S2173, &_S2174);

#line 183
    df_0 _S2176 = _S2172;

#line 183
    df_0 _S2177 = _S2175;

#line 183
    df_0 _S2178 = df_add_0(&_S2176, &_S2177);

#line 183
    df_0 _S2179 = _S504;

#line 183
    df_0 _S2180 = _S437;

#line 183
    df_0 _S2181 = df_mul_0(&_S2179, &_S2180);

#line 183
    df_0 _S2182 = _S437;

#line 183
    df_0 _S2183 = _S504;

#line 183
    df_0 _S2184 = df_mul_0(&_S2182, &_S2183);

#line 183
    df_0 _S2185 = _S2181;

#line 183
    df_0 _S2186 = _S2184;

#line 183
    df_0 _S2187 = df_add_0(&_S2185, &_S2186);

#line 183
    df_0 _S2188 = c2_0;

#line 183
    df_0 _S2189 = _S498;

#line 183
    df_0 _S2190 = df_mul_0(&_S2188, &_S2189);

#line 183
    df_0 _S2191 = _S504;

#line 183
    df_0 _S2192 = _S279;

#line 183
    df_0 _S2193 = df_mul_0(&_S2191, &_S2192);

#line 183
    df_0 _S2194 = _S279;

#line 183
    df_0 _S2195 = _S504;

#line 183
    df_0 _S2196 = df_mul_0(&_S2194, &_S2195);

#line 183
    df_0 _S2197 = _S498;

#line 183
    df_0 _S2198 = c2_0;

#line 183
    df_0 _S2199 = df_mul_0(&_S2197, &_S2198);

#line 183
    df_0 _S2200 = _S2190;

#line 183
    df_0 _S2201 = _S2193;

#line 183
    df_0 _S2202 = df_add_0(&_S2200, &_S2201);

#line 183
    df_0 _S2203 = _S2202;

#line 183
    df_0 _S2204 = _S2196;

#line 183
    df_0 _S2205 = df_add_0(&_S2203, &_S2204);

#line 183
    df_0 _S2206 = _S2205;

#line 183
    df_0 _S2207 = _S2199;

#line 183
    df_0 _S2208 = df_add_0(&_S2206, &_S2207);

#line 183
    df_0 _S2209 = _S504;

#line 183
    df_0 _S2210 = _S282;

#line 183
    df_0 _S2211 = df_mul_0(&_S2209, &_S2210);

#line 183
    df_0 _S2212 = _S282;

#line 183
    df_0 _S2213 = _S504;

#line 183
    df_0 _S2214 = df_mul_0(&_S2212, &_S2213);

#line 183
    df_0 _S2215 = _S2211;

#line 183
    df_0 _S2216 = _S2214;

#line 183
    df_0 _S2217 = df_add_0(&_S2215, &_S2216);

#line 183
    df_0 _S2218 = _S504;

#line 183
    df_0 _S2219 = c1_0;

#line 183
    df_0 _S2220 = df_mul_0(&_S2218, &_S2219);

#line 183
    df_0 _S2221 = c1_0;

#line 183
    df_0 _S2222 = _S504;

#line 183
    df_0 _S2223 = df_mul_0(&_S2221, &_S2222);

#line 183
    df_0 _S2224 = _S2220;

#line 183
    df_0 _S2225 = _S2223;

#line 183
    df_0 _S2226 = df_add_0(&_S2224, &_S2225);

#line 183
    df_0 _S2227 = _S422;

#line 183
    df_0 _S2228 = _S498;

#line 183
    df_0 _S2229 = df_mul_0(&_S2227, &_S2228);

#line 183
    df_0 _S2230 = _S504;

#line 183
    df_0 _S2231 = _S493;

#line 183
    df_0 _S2232 = df_mul_0(&_S2230, &_S2231);

#line 183
    df_0 _S2233 = _S493;

#line 183
    df_0 _S2234 = _S504;

#line 183
    df_0 _S2235 = df_mul_0(&_S2233, &_S2234);

#line 183
    df_0 _S2236 = _S498;

#line 183
    df_0 _S2237 = _S422;

#line 183
    df_0 _S2238 = df_mul_0(&_S2236, &_S2237);

#line 183
    df_0 _S2239 = _S2229;

#line 183
    df_0 _S2240 = _S2232;

#line 183
    df_0 _S2241 = df_add_0(&_S2239, &_S2240);

#line 183
    df_0 _S2242 = _S2241;

#line 183
    df_0 _S2243 = _S2235;

#line 183
    df_0 _S2244 = df_add_0(&_S2242, &_S2243);

#line 183
    df_0 _S2245 = _S2244;

#line 183
    df_0 _S2246 = _S2238;

#line 183
    df_0 _S2247 = df_add_0(&_S2245, &_S2246);

#line 183
    df_0 _S2248 = _S504;

#line 183
    df_0 _S2249 = _S495;

#line 183
    df_0 _S2250 = df_mul_0(&_S2248, &_S2249);

#line 183
    df_0 _S2251 = _S495;

#line 183
    df_0 _S2252 = _S504;

#line 183
    df_0 _S2253 = df_mul_0(&_S2251, &_S2252);

#line 183
    df_0 _S2254 = _S2250;

#line 183
    df_0 _S2255 = _S2253;

#line 183
    df_0 _S2256 = df_add_0(&_S2254, &_S2255);

#line 183
    df_0 _S2257 = _S504;

#line 183
    df_0 _S2258 = _S296;

#line 183
    df_0 _S2259 = df_mul_0(&_S2257, &_S2258);

#line 183
    df_0 _S2260 = _S296;

#line 183
    df_0 _S2261 = _S504;

#line 183
    df_0 _S2262 = df_mul_0(&_S2260, &_S2261);

#line 183
    df_0 _S2263 = _S2259;

#line 183
    df_0 _S2264 = _S2262;

#line 183
    df_0 _S2265 = df_add_0(&_S2263, &_S2264);

#line 183
    df_0 _S2266 = _S428;

#line 183
    df_0 _S2267 = _S507;

#line 183
    df_0 _S2268 = df_mul_0(&_S2266, &_S2267);

#line 183
    df_0 _S2269 = _S507;

#line 183
    df_0 _S2270 = _S428;

#line 183
    df_0 _S2271 = df_mul_0(&_S2269, &_S2270);

#line 183
    df_0 _S2272 = _S2268;

#line 183
    df_0 _S2273 = _S2271;

#line 183
    df_0 _S2274 = df_add_0(&_S2272, &_S2273);

#line 183
    df_0 _S2275 = _S428;

#line 183
    df_0 _S2276 = _S510;

#line 183
    df_0 _S2277 = df_mul_0(&_S2275, &_S2276);

#line 183
    df_0 _S2278 = _S510;

#line 183
    df_0 _S2279 = _S428;

#line 183
    df_0 _S2280 = df_mul_0(&_S2278, &_S2279);

#line 183
    df_0 _S2281 = _S2277;

#line 183
    df_0 _S2282 = _S2280;

#line 183
    df_0 _S2283 = df_add_0(&_S2281, &_S2282);

#line 183
    df_0 _S2284 = _S428;

#line 183
    df_0 _S2285 = _S279;

#line 183
    df_0 _S2286 = df_mul_0(&_S2284, &_S2285);

#line 183
    df_0 _S2287 = _S279;

#line 183
    df_0 _S2288 = _S428;

#line 183
    df_0 _S2289 = df_mul_0(&_S2287, &_S2288);

#line 183
    df_0 _S2290 = _S2286;

#line 183
    df_0 _S2291 = _S2289;

#line 183
    df_0 _S2292 = df_add_0(&_S2290, &_S2291);

#line 183
    df_0 _S2293 = _S428;

#line 183
    df_0 _S2294 = _S282;

#line 183
    df_0 _S2295 = df_mul_0(&_S2293, &_S2294);

#line 183
    df_0 _S2296 = _S282;

#line 183
    df_0 _S2297 = _S428;

#line 183
    df_0 _S2298 = df_mul_0(&_S2296, &_S2297);

#line 183
    df_0 _S2299 = _S2295;

#line 183
    df_0 _S2300 = _S2298;

#line 183
    df_0 _S2301 = df_add_0(&_S2299, &_S2300);

#line 183
    df_0 _S2302 = _S428;

#line 183
    df_0 _S2303 = _S493;

#line 183
    df_0 _S2304 = df_mul_0(&_S2302, &_S2303);

#line 183
    df_0 _S2305 = _S493;

#line 183
    df_0 _S2306 = _S428;

#line 183
    df_0 _S2307 = df_mul_0(&_S2305, &_S2306);

#line 183
    df_0 _S2308 = _S2304;

#line 183
    df_0 _S2309 = _S2307;

#line 183
    df_0 _S2310 = df_add_0(&_S2308, &_S2309);

#line 183
    df_0 _S2311 = _S428;

#line 183
    df_0 _S2312 = _S495;

#line 183
    df_0 _S2313 = df_mul_0(&_S2311, &_S2312);

#line 183
    df_0 _S2314 = _S495;

#line 183
    df_0 _S2315 = _S428;

#line 183
    df_0 _S2316 = df_mul_0(&_S2314, &_S2315);

#line 183
    df_0 _S2317 = _S2313;

#line 183
    df_0 _S2318 = _S2316;

#line 183
    df_0 _S2319 = df_add_0(&_S2317, &_S2318);

#line 183
    df_0 _S2320 = _S507;

#line 183
    df_0 _S2321 = _S507;

#line 183
    df_0 _S2322 = df_mul_0(&_S2320, &_S2321);

#line 183
    df_0 _S2323 = _S2322;

#line 183
    df_0 _S2324 = _S2322;

#line 183
    df_0 _S2325 = df_add_0(&_S2323, &_S2324);

#line 183
    df_0 _S2326 = _S507;

#line 183
    df_0 _S2327 = _S510;

#line 183
    df_0 _S2328 = df_mul_0(&_S2326, &_S2327);

#line 183
    df_0 _S2329 = _S510;

#line 183
    df_0 _S2330 = _S507;

#line 183
    df_0 _S2331 = df_mul_0(&_S2329, &_S2330);

#line 183
    df_0 _S2332 = _S2328;

#line 183
    df_0 _S2333 = _S2331;

#line 183
    df_0 _S2334 = df_add_0(&_S2332, &_S2333);

#line 183
    df_0 _S2335 = _S507;

#line 183
    df_0 _S2336 = _S437;

#line 183
    df_0 _S2337 = df_mul_0(&_S2335, &_S2336);

#line 183
    df_0 _S2338 = _S437;

#line 183
    df_0 _S2339 = _S507;

#line 183
    df_0 _S2340 = df_mul_0(&_S2338, &_S2339);

#line 183
    df_0 _S2341 = _S2337;

#line 183
    df_0 _S2342 = _S2340;

#line 183
    df_0 _S2343 = df_add_0(&_S2341, &_S2342);

#line 183
    df_0 _S2344 = _S507;

#line 183
    df_0 _S2345 = _S279;

#line 183
    df_0 _S2346 = df_mul_0(&_S2344, &_S2345);

#line 183
    df_0 _S2347 = _S279;

#line 183
    df_0 _S2348 = _S507;

#line 183
    df_0 _S2349 = df_mul_0(&_S2347, &_S2348);

#line 183
    df_0 _S2350 = _S2346;

#line 183
    df_0 _S2351 = _S2349;

#line 183
    df_0 _S2352 = df_add_0(&_S2350, &_S2351);

#line 183
    df_0 _S2353 = _S507;

#line 183
    df_0 _S2354 = _S282;

#line 183
    df_0 _S2355 = df_mul_0(&_S2353, &_S2354);

#line 183
    df_0 _S2356 = _S282;

#line 183
    df_0 _S2357 = _S507;

#line 183
    df_0 _S2358 = df_mul_0(&_S2356, &_S2357);

#line 183
    df_0 _S2359 = _S2190;

#line 183
    df_0 _S2360 = _S2355;

#line 183
    df_0 _S2361 = df_add_0(&_S2359, &_S2360);

#line 183
    df_0 _S2362 = _S2361;

#line 183
    df_0 _S2363 = _S2358;

#line 183
    df_0 _S2364 = df_add_0(&_S2362, &_S2363);

#line 183
    df_0 _S2365 = _S2364;

#line 183
    df_0 _S2366 = _S2199;

#line 183
    df_0 _S2367 = df_add_0(&_S2365, &_S2366);

#line 183
    df_0 _S2368 = _S507;

#line 183
    df_0 _S2369 = c1_0;

#line 183
    df_0 _S2370 = df_mul_0(&_S2368, &_S2369);

#line 183
    df_0 _S2371 = c1_0;

#line 183
    df_0 _S2372 = _S507;

#line 183
    df_0 _S2373 = df_mul_0(&_S2371, &_S2372);

#line 183
    df_0 _S2374 = _S2370;

#line 183
    df_0 _S2375 = _S2373;

#line 183
    df_0 _S2376 = df_add_0(&_S2374, &_S2375);

#line 183
    df_0 _S2377 = _S507;

#line 183
    df_0 _S2378 = _S493;

#line 183
    df_0 _S2379 = df_mul_0(&_S2377, &_S2378);

#line 183
    df_0 _S2380 = _S493;

#line 183
    df_0 _S2381 = _S507;

#line 183
    df_0 _S2382 = df_mul_0(&_S2380, &_S2381);

#line 183
    df_0 _S2383 = _S2379;

#line 183
    df_0 _S2384 = _S2382;

#line 183
    df_0 _S2385 = df_add_0(&_S2383, &_S2384);

#line 183
    df_0 _S2386 = _S507;

#line 183
    df_0 _S2387 = _S495;

#line 183
    df_0 _S2388 = df_mul_0(&_S2386, &_S2387);

#line 183
    df_0 _S2389 = _S495;

#line 183
    df_0 _S2390 = _S507;

#line 183
    df_0 _S2391 = df_mul_0(&_S2389, &_S2390);

#line 183
    df_0 _S2392 = _S2229;

#line 183
    df_0 _S2393 = _S2388;

#line 183
    df_0 _S2394 = df_add_0(&_S2392, &_S2393);

#line 183
    df_0 _S2395 = _S2394;

#line 183
    df_0 _S2396 = _S2391;

#line 183
    df_0 _S2397 = df_add_0(&_S2395, &_S2396);

#line 183
    df_0 _S2398 = _S2397;

#line 183
    df_0 _S2399 = _S2238;

#line 183
    df_0 _S2400 = df_add_0(&_S2398, &_S2399);

#line 183
    df_0 _S2401 = _S507;

#line 183
    df_0 _S2402 = _S296;

#line 183
    df_0 _S2403 = df_mul_0(&_S2401, &_S2402);

#line 183
    df_0 _S2404 = _S296;

#line 183
    df_0 _S2405 = _S507;

#line 183
    df_0 _S2406 = df_mul_0(&_S2404, &_S2405);

#line 183
    df_0 _S2407 = _S2403;

#line 183
    df_0 _S2408 = _S2406;

#line 183
    df_0 _S2409 = df_add_0(&_S2407, &_S2408);

#line 183
    df_0 _S2410 = _S510;

#line 183
    df_0 _S2411 = _S510;

#line 183
    df_0 _S2412 = df_mul_0(&_S2410, &_S2411);

#line 183
    df_0 _S2413 = _S2412;

#line 183
    df_0 _S2414 = _S2412;

#line 183
    df_0 _S2415 = df_add_0(&_S2413, &_S2414);

#line 183
    df_0 _S2416 = _S510;

#line 183
    df_0 _S2417 = _S437;

#line 183
    df_0 _S2418 = df_mul_0(&_S2416, &_S2417);

#line 183
    df_0 _S2419 = _S437;

#line 183
    df_0 _S2420 = _S510;

#line 183
    df_0 _S2421 = df_mul_0(&_S2419, &_S2420);

#line 183
    df_0 _S2422 = _S2418;

#line 183
    df_0 _S2423 = _S2421;

#line 183
    df_0 _S2424 = df_add_0(&_S2422, &_S2423);

#line 183
    df_0 _S2425 = _S510;

#line 183
    df_0 _S2426 = _S279;

#line 183
    df_0 _S2427 = df_mul_0(&_S2425, &_S2426);

#line 183
    df_0 _S2428 = _S279;

#line 183
    df_0 _S2429 = _S510;

#line 183
    df_0 _S2430 = df_mul_0(&_S2428, &_S2429);

#line 183
    df_0 _S2431 = _S2067;

#line 183
    df_0 _S2432 = _S2427;

#line 183
    df_0 _S2433 = df_add_0(&_S2431, &_S2432);

#line 183
    df_0 _S2434 = _S2433;

#line 183
    df_0 _S2435 = _S2430;

#line 183
    df_0 _S2436 = df_add_0(&_S2434, &_S2435);

#line 183
    df_0 _S2437 = _S2436;

#line 183
    df_0 _S2438 = _S2076;

#line 183
    df_0 _S2439 = df_add_0(&_S2437, &_S2438);

#line 183
    df_0 _S2440 = _S510;

#line 183
    df_0 _S2441 = _S282;

#line 183
    df_0 _S2442 = df_mul_0(&_S2440, &_S2441);

#line 183
    df_0 _S2443 = _S282;

#line 183
    df_0 _S2444 = _S510;

#line 183
    df_0 _S2445 = df_mul_0(&_S2443, &_S2444);

#line 183
    df_0 _S2446 = _S2442;

#line 183
    df_0 _S2447 = _S2445;

#line 183
    df_0 _S2448 = df_add_0(&_S2446, &_S2447);

#line 183
    df_0 _S2449 = _S510;

#line 183
    df_0 _S2450 = c1_0;

#line 183
    df_0 _S2451 = df_mul_0(&_S2449, &_S2450);

#line 183
    df_0 _S2452 = c1_0;

#line 183
    df_0 _S2453 = _S510;

#line 183
    df_0 _S2454 = df_mul_0(&_S2452, &_S2453);

#line 183
    df_0 _S2455 = _S2451;

#line 183
    df_0 _S2456 = _S2454;

#line 183
    df_0 _S2457 = df_add_0(&_S2455, &_S2456);

#line 183
    df_0 _S2458 = _S510;

#line 183
    df_0 _S2459 = _S493;

#line 183
    df_0 _S2460 = df_mul_0(&_S2458, &_S2459);

#line 183
    df_0 _S2461 = _S493;

#line 183
    df_0 _S2462 = _S510;

#line 183
    df_0 _S2463 = df_mul_0(&_S2461, &_S2462);

#line 183
    df_0 _S2464 = _S2106;

#line 183
    df_0 _S2465 = _S2460;

#line 183
    df_0 _S2466 = df_add_0(&_S2464, &_S2465);

#line 183
    df_0 _S2467 = _S2466;

#line 183
    df_0 _S2468 = _S2463;

#line 183
    df_0 _S2469 = df_add_0(&_S2467, &_S2468);

#line 183
    df_0 _S2470 = _S2469;

#line 183
    df_0 _S2471 = _S2115;

#line 183
    df_0 _S2472 = df_add_0(&_S2470, &_S2471);

#line 183
    df_0 _S2473 = _S510;

#line 183
    df_0 _S2474 = _S495;

#line 183
    df_0 _S2475 = df_mul_0(&_S2473, &_S2474);

#line 183
    df_0 _S2476 = _S495;

#line 183
    df_0 _S2477 = _S510;

#line 183
    df_0 _S2478 = df_mul_0(&_S2476, &_S2477);

#line 183
    df_0 _S2479 = _S2475;

#line 183
    df_0 _S2480 = _S2478;

#line 183
    df_0 _S2481 = df_add_0(&_S2479, &_S2480);

#line 183
    df_0 _S2482 = _S510;

#line 183
    df_0 _S2483 = _S296;

#line 183
    df_0 _S2484 = df_mul_0(&_S2482, &_S2483);

#line 183
    df_0 _S2485 = _S296;

#line 183
    df_0 _S2486 = _S510;

#line 183
    df_0 _S2487 = df_mul_0(&_S2485, &_S2486);

#line 183
    df_0 _S2488 = _S2484;

#line 183
    df_0 _S2489 = _S2487;

#line 183
    df_0 _S2490 = df_add_0(&_S2488, &_S2489);

#line 183
    df_0 _S2491 = _S437;

#line 183
    df_0 _S2492 = _S279;

#line 183
    df_0 _S2493 = df_mul_0(&_S2491, &_S2492);

#line 183
    df_0 _S2494 = _S279;

#line 183
    df_0 _S2495 = _S437;

#line 183
    df_0 _S2496 = df_mul_0(&_S2494, &_S2495);

#line 183
    df_0 _S2497 = _S2493;

#line 183
    df_0 _S2498 = _S2496;

#line 183
    df_0 _S2499 = df_add_0(&_S2497, &_S2498);

#line 183
    df_0 _S2500 = _S437;

#line 183
    df_0 _S2501 = _S282;

#line 183
    df_0 _S2502 = df_mul_0(&_S2500, &_S2501);

#line 183
    df_0 _S2503 = _S282;

#line 183
    df_0 _S2504 = _S437;

#line 183
    df_0 _S2505 = df_mul_0(&_S2503, &_S2504);

#line 183
    df_0 _S2506 = _S2502;

#line 183
    df_0 _S2507 = _S2505;

#line 183
    df_0 _S2508 = df_add_0(&_S2506, &_S2507);

#line 183
    df_0 _S2509 = _S437;

#line 183
    df_0 _S2510 = _S493;

#line 183
    df_0 _S2511 = df_mul_0(&_S2509, &_S2510);

#line 183
    df_0 _S2512 = _S493;

#line 183
    df_0 _S2513 = _S437;

#line 183
    df_0 _S2514 = df_mul_0(&_S2512, &_S2513);

#line 183
    df_0 _S2515 = _S2511;

#line 183
    df_0 _S2516 = _S2514;

#line 183
    df_0 _S2517 = df_add_0(&_S2515, &_S2516);

#line 183
    df_0 _S2518 = _S437;

#line 183
    df_0 _S2519 = _S495;

#line 183
    df_0 _S2520 = df_mul_0(&_S2518, &_S2519);

#line 183
    df_0 _S2521 = _S495;

#line 183
    df_0 _S2522 = _S437;

#line 183
    df_0 _S2523 = df_mul_0(&_S2521, &_S2522);

#line 183
    df_0 _S2524 = _S2520;

#line 183
    df_0 _S2525 = _S2523;

#line 183
    df_0 _S2526 = df_add_0(&_S2524, &_S2525);

#line 183
    df_0 _S2527 = _S279;

#line 183
    df_0 _S2528 = _S279;

#line 183
    df_0 _S2529 = df_mul_0(&_S2527, &_S2528);

#line 183
    df_0 _S2530 = _S2529;

#line 183
    df_0 _S2531 = _S2529;

#line 183
    df_0 _S2532 = df_add_0(&_S2530, &_S2531);

#line 183
    df_0 _S2533 = _S279;

#line 183
    df_0 _S2534 = _S282;

#line 183
    df_0 _S2535 = df_mul_0(&_S2533, &_S2534);

#line 183
    df_0 _S2536 = _S282;

#line 183
    df_0 _S2537 = _S279;

#line 183
    df_0 _S2538 = df_mul_0(&_S2536, &_S2537);

#line 183
    df_0 _S2539 = _S2535;

#line 183
    df_0 _S2540 = _S2538;

#line 183
    df_0 _S2541 = df_add_0(&_S2539, &_S2540);

#line 183
    df_0 _S2542 = _S279;

#line 183
    df_0 _S2543 = c1_0;

#line 183
    df_0 _S2544 = df_mul_0(&_S2542, &_S2543);

#line 183
    df_0 _S2545 = c1_0;

#line 183
    df_0 _S2546 = _S279;

#line 183
    df_0 _S2547 = df_mul_0(&_S2545, &_S2546);

#line 183
    df_0 _S2548 = _S2544;

#line 183
    df_0 _S2549 = _S2547;

#line 183
    df_0 _S2550 = df_add_0(&_S2548, &_S2549);

#line 183
    df_0 _S2551 = _S279;

#line 183
    df_0 _S2552 = _S493;

#line 183
    df_0 _S2553 = df_mul_0(&_S2551, &_S2552);

#line 183
    df_0 _S2554 = _S493;

#line 183
    df_0 _S2555 = _S279;

#line 183
    df_0 _S2556 = df_mul_0(&_S2554, &_S2555);

#line 183
    df_0 _S2557 = _S2553;

#line 183
    df_0 _S2558 = _S2556;

#line 183
    df_0 _S2559 = df_add_0(&_S2557, &_S2558);

#line 183
    df_0 _S2560 = _S279;

#line 183
    df_0 _S2561 = _S495;

#line 183
    df_0 _S2562 = df_mul_0(&_S2560, &_S2561);

#line 183
    df_0 _S2563 = _S495;

#line 183
    df_0 _S2564 = _S279;

#line 183
    df_0 _S2565 = df_mul_0(&_S2563, &_S2564);

#line 183
    df_0 _S2566 = _S2562;

#line 183
    df_0 _S2567 = _S2565;

#line 183
    df_0 _S2568 = df_add_0(&_S2566, &_S2567);

#line 183
    df_0 _S2569 = _S279;

#line 183
    df_0 _S2570 = _S296;

#line 183
    df_0 _S2571 = df_mul_0(&_S2569, &_S2570);

#line 183
    df_0 _S2572 = _S296;

#line 183
    df_0 _S2573 = _S279;

#line 183
    df_0 _S2574 = df_mul_0(&_S2572, &_S2573);

#line 183
    df_0 _S2575 = _S2571;

#line 183
    df_0 _S2576 = _S2574;

#line 183
    df_0 _S2577 = df_add_0(&_S2575, &_S2576);

#line 183
    df_0 _S2578 = _S282;

#line 183
    df_0 _S2579 = _S282;

#line 183
    df_0 _S2580 = df_mul_0(&_S2578, &_S2579);

#line 183
    df_0 _S2581 = _S2580;

#line 183
    df_0 _S2582 = _S2580;

#line 183
    df_0 _S2583 = df_add_0(&_S2581, &_S2582);

#line 183
    df_0 _S2584 = _S282;

#line 183
    df_0 _S2585 = c1_0;

#line 183
    df_0 _S2586 = df_mul_0(&_S2584, &_S2585);

#line 183
    df_0 _S2587 = c1_0;

#line 183
    df_0 _S2588 = _S282;

#line 183
    df_0 _S2589 = df_mul_0(&_S2587, &_S2588);

#line 183
    df_0 _S2590 = _S2586;

#line 183
    df_0 _S2591 = _S2589;

#line 183
    df_0 _S2592 = df_add_0(&_S2590, &_S2591);

#line 183
    df_0 _S2593 = _S282;

#line 183
    df_0 _S2594 = _S493;

#line 183
    df_0 _S2595 = df_mul_0(&_S2593, &_S2594);

#line 183
    df_0 _S2596 = _S493;

#line 183
    df_0 _S2597 = _S282;

#line 183
    df_0 _S2598 = df_mul_0(&_S2596, &_S2597);

#line 183
    df_0 _S2599 = _S2595;

#line 183
    df_0 _S2600 = _S2598;

#line 183
    df_0 _S2601 = df_add_0(&_S2599, &_S2600);

#line 183
    df_0 _S2602 = _S282;

#line 183
    df_0 _S2603 = _S495;

#line 183
    df_0 _S2604 = df_mul_0(&_S2602, &_S2603);

#line 183
    df_0 _S2605 = _S495;

#line 183
    df_0 _S2606 = _S282;

#line 183
    df_0 _S2607 = df_mul_0(&_S2605, &_S2606);

#line 183
    df_0 _S2608 = _S2604;

#line 183
    df_0 _S2609 = _S2607;

#line 183
    df_0 _S2610 = df_add_0(&_S2608, &_S2609);

#line 183
    df_0 _S2611 = _S282;

#line 183
    df_0 _S2612 = _S296;

#line 183
    df_0 _S2613 = df_mul_0(&_S2611, &_S2612);

#line 183
    df_0 _S2614 = _S296;

#line 183
    df_0 _S2615 = _S282;

#line 183
    df_0 _S2616 = df_mul_0(&_S2614, &_S2615);

#line 183
    df_0 _S2617 = _S2613;

#line 183
    df_0 _S2618 = _S2616;

#line 183
    df_0 _S2619 = df_add_0(&_S2617, &_S2618);

#line 183
    df_0 _S2620 = c1_0;

#line 183
    df_0 _S2621 = _S493;

#line 183
    df_0 _S2622 = df_mul_0(&_S2620, &_S2621);

#line 183
    df_0 _S2623 = _S493;

#line 183
    df_0 _S2624 = c1_0;

#line 183
    df_0 _S2625 = df_mul_0(&_S2623, &_S2624);

#line 183
    df_0 _S2626 = _S2622;

#line 183
    df_0 _S2627 = _S2625;

#line 183
    df_0 _S2628 = df_add_0(&_S2626, &_S2627);

#line 183
    df_0 _S2629 = c1_0;

#line 183
    df_0 _S2630 = _S495;

#line 183
    df_0 _S2631 = df_mul_0(&_S2629, &_S2630);

#line 183
    df_0 _S2632 = _S495;

#line 183
    df_0 _S2633 = c1_0;

#line 183
    df_0 _S2634 = df_mul_0(&_S2632, &_S2633);

#line 183
    df_0 _S2635 = _S2631;

#line 183
    df_0 _S2636 = _S2634;

#line 183
    df_0 _S2637 = df_add_0(&_S2635, &_S2636);

#line 183
    df_0 _S2638 = _S493;

#line 183
    df_0 _S2639 = _S493;

#line 183
    df_0 _S2640 = df_mul_0(&_S2638, &_S2639);

#line 183
    df_0 _S2641 = _S2640;

#line 183
    df_0 _S2642 = _S2640;

#line 183
    df_0 _S2643 = df_add_0(&_S2641, &_S2642);

#line 183
    df_0 _S2644 = _S493;

#line 183
    df_0 _S2645 = _S495;

#line 183
    df_0 _S2646 = df_mul_0(&_S2644, &_S2645);

#line 183
    df_0 _S2647 = _S495;

#line 183
    df_0 _S2648 = _S493;

#line 183
    df_0 _S2649 = df_mul_0(&_S2647, &_S2648);

#line 183
    df_0 _S2650 = _S2646;

#line 183
    df_0 _S2651 = _S2649;

#line 183
    df_0 _S2652 = df_add_0(&_S2650, &_S2651);

#line 183
    df_0 _S2653 = _S493;

#line 183
    df_0 _S2654 = _S296;

#line 183
    df_0 _S2655 = df_mul_0(&_S2653, &_S2654);

#line 183
    df_0 _S2656 = _S296;

#line 183
    df_0 _S2657 = _S493;

#line 183
    df_0 _S2658 = df_mul_0(&_S2656, &_S2657);

#line 183
    df_0 _S2659 = _S2655;

#line 183
    df_0 _S2660 = _S2658;

#line 183
    df_0 _S2661 = df_add_0(&_S2659, &_S2660);

#line 183
    df_0 _S2662 = _S495;

#line 183
    df_0 _S2663 = _S495;

#line 183
    df_0 _S2664 = df_mul_0(&_S2662, &_S2663);

#line 183
    df_0 _S2665 = _S2664;

#line 183
    df_0 _S2666 = _S2664;

#line 183
    df_0 _S2667 = df_add_0(&_S2665, &_S2666);

#line 183
    df_0 _S2668 = _S495;

#line 183
    df_0 _S2669 = _S296;

#line 183
    df_0 _S2670 = df_mul_0(&_S2668, &_S2669);

#line 183
    df_0 _S2671 = _S296;

#line 183
    df_0 _S2672 = _S495;

#line 183
    df_0 _S2673 = df_mul_0(&_S2671, &_S2672);

#line 183
    df_0 _S2674 = _S2670;

#line 183
    df_0 _S2675 = _S2673;

#line 183
    df_0 _S2676 = df_add_0(&_S2674, &_S2675);

#line 183
    df_0 _S2677 = _S516;

#line 183
    df_0 _S2678 = _S1290;

#line 183
    df_0 _S2679 = df_add_0(&_S2677, &_S2678);

#line 183
    df_0 _S2680 = _S525;

#line 183
    df_0 _S2681 = _S1299;

#line 183
    df_0 _S2682 = df_add_0(&_S2680, &_S2681);

#line 183
    df_0 _S2683 = _S534;

#line 183
    df_0 _S2684 = _S1320;

#line 183
    df_0 _S2685 = df_add_0(&_S2683, &_S2684);

#line 183
    df_0 _S2686 = _S543;

#line 183
    df_0 _S2687 = _S1329;

#line 183
    df_0 _S2688 = df_add_0(&_S2686, &_S2687);

#line 183
    df_0 _S2689 = _S552;

#line 183
    df_0 _S2690 = _S1338;

#line 183
    df_0 _S2691 = df_add_0(&_S2689, &_S2690);

#line 183
    df_0 _S2692 = _S561;

#line 183
    df_0 _S2693 = _S1359;

#line 183
    df_0 _S2694 = df_add_0(&_S2692, &_S2693);

#line 183
    df_0 _S2695 = _S570;

#line 183
    df_0 _S2696 = _S1368;

#line 183
    df_0 _S2697 = df_add_0(&_S2695, &_S2696);

#line 183
    df_0 _S2698 = _S579;

#line 183
    df_0 _S2699 = _S1377;

#line 183
    df_0 _S2700 = df_add_0(&_S2698, &_S2699);

#line 183
    df_0 _S2701 = _S588;

#line 183
    df_0 _S2702 = _S1398;

#line 183
    df_0 _S2703 = df_add_0(&_S2701, &_S2702);

#line 183
    df_0 _S2704 = _S597;

#line 183
    df_0 _S2705 = _S1407;

#line 183
    df_0 _S2706 = df_add_0(&_S2704, &_S2705);

#line 183
    df_0 _S2707 = _S606;

#line 183
    df_0 _S2708 = _S1416;

#line 183
    df_0 _S2709 = df_add_0(&_S2707, &_S2708);

#line 183
    df_0 _S2710 = _S615;

#line 183
    df_0 _S2711 = _S1437;

#line 183
    df_0 _S2712 = df_add_0(&_S2710, &_S2711);

#line 183
    df_0 _S2713 = _S621;

#line 183
    df_0 _S2714 = _S516;

#line 183
    df_0 _S2715 = df_add_0(&_S2713, &_S2714);

#line 183
    df_0 _S2716 = _S642;

#line 183
    df_0 _S2717 = _S1446;

#line 183
    df_0 _S2718 = df_add_0(&_S2716, &_S2717);

#line 183
    df_0 _S2719 = _S651;

#line 183
    df_0 _S2720 = _S1455;

#line 183
    df_0 _S2721 = df_add_0(&_S2719, &_S2720);

#line 183
    df_0 _S2722 = _S660;

#line 183
    df_0 _S2723 = _S543;

#line 183
    df_0 _S2724 = df_add_0(&_S2722, &_S2723);

#line 183
    df_0 _S2725 = _S681;

#line 183
    df_0 _S2726 = _S1464;

#line 183
    df_0 _S2727 = df_add_0(&_S2725, &_S2726);

#line 183
    df_0 _S2728 = _S690;

#line 183
    df_0 _S2729 = _S1473;

#line 183
    df_0 _S2730 = df_add_0(&_S2728, &_S2729);

#line 183
    df_0 _S2731 = _S699;

#line 183
    df_0 _S2732 = _S570;

#line 183
    df_0 _S2733 = df_add_0(&_S2731, &_S2732);

#line 183
    df_0 _S2734 = _S720;

#line 183
    df_0 _S2735 = _S1482;

#line 183
    df_0 _S2736 = df_add_0(&_S2734, &_S2735);

#line 183
    df_0 _S2737 = _S729;

#line 183
    df_0 _S2738 = _S1491;

#line 183
    df_0 _S2739 = df_add_0(&_S2737, &_S2738);

#line 183
    df_0 _S2740 = _S738;

#line 183
    df_0 _S2741 = _S597;

#line 183
    df_0 _S2742 = df_add_0(&_S2740, &_S2741);

#line 183
    df_0 _S2743 = _S759;

#line 183
    df_0 _S2744 = _S1500;

#line 183
    df_0 _S2745 = df_add_0(&_S2743, &_S2744);

#line 183
    df_0 _S2746 = _S765;

#line 183
    df_0 _S2747 = _S1506;

#line 183
    df_0 _S2748 = df_add_0(&_S2746, &_S2747);

#line 183
    df_0 _S2749 = _S774;

#line 183
    df_0 _S2750 = _S1527;

#line 183
    df_0 _S2751 = df_add_0(&_S2749, &_S2750);

#line 183
    df_0 _S2752 = _S795;

#line 183
    df_0 _S2753 = _S1536;

#line 183
    df_0 _S2754 = df_add_0(&_S2752, &_S2753);

#line 183
    df_0 _S2755 = _S804;

#line 183
    df_0 _S2756 = _S1545;

#line 183
    df_0 _S2757 = df_add_0(&_S2755, &_S2756);

#line 183
    df_0 _S2758 = _S813;

#line 183
    df_0 _S2759 = _S1566;

#line 183
    df_0 _S2760 = df_add_0(&_S2758, &_S2759);

#line 183
    df_0 _S2761 = _S834;

#line 183
    df_0 _S2762 = _S1575;

#line 183
    df_0 _S2763 = df_add_0(&_S2761, &_S2762);

#line 183
    df_0 _S2764 = _S843;

#line 183
    df_0 _S2765 = _S1584;

#line 183
    df_0 _S2766 = df_add_0(&_S2764, &_S2765);

#line 183
    df_0 _S2767 = _S852;

#line 183
    df_0 _S2768 = _S1605;

#line 183
    df_0 _S2769 = df_add_0(&_S2767, &_S2768);

#line 183
    df_0 _S2770 = _S873;

#line 183
    df_0 _S2771 = _S1614;

#line 183
    df_0 _S2772 = df_add_0(&_S2770, &_S2771);

#line 183
    df_0 _S2773 = _S882;

#line 183
    df_0 _S2774 = _S1623;

#line 183
    df_0 _S2775 = df_add_0(&_S2773, &_S2774);

#line 183
    df_0 _S2776 = _S888;

#line 183
    df_0 _S2777 = _S1629;

#line 183
    df_0 _S2778 = df_add_0(&_S2776, &_S2777);

#line 183
    df_0 _S2779 = _S897;

#line 183
    df_0 _S2780 = _S1638;

#line 183
    df_0 _S2781 = df_add_0(&_S2779, &_S2780);

#line 183
    df_0 _S2782 = _S906;

#line 183
    df_0 _S2783 = _S1647;

#line 183
    df_0 _S2784 = df_add_0(&_S2782, &_S2783);

#line 183
    df_0 _S2785 = _S915;

#line 183
    df_0 _S2786 = _S1656;

#line 183
    df_0 _S2787 = df_add_0(&_S2785, &_S2786);

#line 183
    df_0 _S2788 = _S924;

#line 183
    df_0 _S2789 = _S1665;

#line 183
    df_0 _S2790 = df_add_0(&_S2788, &_S2789);

#line 183
    df_0 _S2791 = _S933;

#line 183
    df_0 _S2792 = _S1680;

#line 183
    df_0 _S2793 = df_add_0(&_S2791, &_S2792);

#line 183
    df_0 _S2794 = _S942;

#line 183
    df_0 _S2795 = _S1689;

#line 183
    df_0 _S2796 = df_add_0(&_S2794, &_S2795);

#line 183
    df_0 _S2797 = _S951;

#line 183
    df_0 _S2798 = _S1698;

#line 183
    df_0 _S2799 = df_add_0(&_S2797, &_S2798);

#line 183
    df_0 _S2800 = _S960;

#line 183
    df_0 _S2801 = _S1713;

#line 183
    df_0 _S2802 = df_add_0(&_S2800, &_S2801);

#line 183
    df_0 _S2803 = _S966;

#line 183
    df_0 _S2804 = _S888;

#line 183
    df_0 _S2805 = df_add_0(&_S2803, &_S2804);

#line 183
    df_0 _S2806 = _S975;

#line 183
    df_0 _S2807 = _S1722;

#line 183
    df_0 _S2808 = df_add_0(&_S2806, &_S2807);

#line 183
    df_0 _S2809 = _S984;

#line 183
    df_0 _S2810 = _S1731;

#line 183
    df_0 _S2811 = df_add_0(&_S2809, &_S2810);

#line 183
    df_0 _S2812 = _S993;

#line 183
    df_0 _S2813 = _S915;

#line 183
    df_0 _S2814 = df_add_0(&_S2812, &_S2813);

#line 183
    df_0 _S2815 = _S1008;

#line 183
    df_0 _S2816 = _S1740;

#line 183
    df_0 _S2817 = df_add_0(&_S2815, &_S2816);

#line 183
    df_0 _S2818 = _S1017;

#line 183
    df_0 _S2819 = _S1749;

#line 183
    df_0 _S2820 = df_add_0(&_S2818, &_S2819);

#line 183
    df_0 _S2821 = _S1026;

#line 183
    df_0 _S2822 = _S942;

#line 183
    df_0 _S2823 = df_add_0(&_S2821, &_S2822);

#line 183
    df_0 _S2824 = _S1041;

#line 183
    df_0 _S2825 = _S1758;

#line 183
    df_0 _S2826 = df_add_0(&_S2824, &_S2825);

#line 183
    df_0 _S2827 = _S1047;

#line 183
    df_0 _S2828 = _S1764;

#line 183
    df_0 _S2829 = df_add_0(&_S2827, &_S2828);

#line 183
    df_0 _S2830 = _S1056;

#line 183
    df_0 _S2831 = _S1779;

#line 183
    df_0 _S2832 = df_add_0(&_S2830, &_S2831);

#line 183
    df_0 _S2833 = _S1071;

#line 183
    df_0 _S2834 = _S1788;

#line 183
    df_0 _S2835 = df_add_0(&_S2833, &_S2834);

#line 183
    df_0 _S2836 = _S1080;

#line 183
    df_0 _S2837 = _S1797;

#line 183
    df_0 _S2838 = df_add_0(&_S2836, &_S2837);

#line 183
    df_0 _S2839 = _S1089;

#line 183
    df_0 _S2840 = _S1812;

#line 183
    df_0 _S2841 = df_add_0(&_S2839, &_S2840);

#line 183
    df_0 _S2842 = _S1104;

#line 183
    df_0 _S2843 = _S1821;

#line 183
    df_0 _S2844 = df_add_0(&_S2842, &_S2843);

#line 183
    df_0 _S2845 = _S1113;

#line 183
    df_0 _S2846 = _S1830;

#line 183
    df_0 _S2847 = df_add_0(&_S2845, &_S2846);

#line 183
    df_0 _S2848 = _S1119;

#line 183
    df_0 _S2849 = _S1836;

#line 183
    df_0 _S2850 = df_add_0(&_S2848, &_S2849);

#line 183
    df_0 _S2851 = _S1128;

#line 183
    df_0 _S2852 = _S1845;

#line 183
    df_0 _S2853 = df_add_0(&_S2851, &_S2852);

#line 183
    df_0 _S2854 = _S1137;

#line 183
    df_0 _S2855 = _S1854;

#line 183
    df_0 _S2856 = df_add_0(&_S2854, &_S2855);

#line 183
    df_0 _S2857 = _S1146;

#line 183
    df_0 _S2858 = _S1863;

#line 183
    df_0 _S2859 = df_add_0(&_S2857, &_S2858);

#line 183
    df_0 _S2860 = _S1155;

#line 183
    df_0 _S2861 = _S1872;

#line 183
    df_0 _S2862 = df_add_0(&_S2860, &_S2861);

#line 183
    df_0 _S2863 = _S1164;

#line 183
    df_0 _S2864 = _S1881;

#line 183
    df_0 _S2865 = df_add_0(&_S2863, &_S2864);

#line 183
    df_0 _S2866 = _S1170;

#line 183
    df_0 _S2867 = _S1119;

#line 183
    df_0 _S2868 = df_add_0(&_S2866, &_S2867);

#line 183
    df_0 _S2869 = _S1179;

#line 183
    df_0 _S2870 = _S1890;

#line 183
    df_0 _S2871 = df_add_0(&_S2869, &_S2870);

#line 183
    df_0 _S2872 = _S1188;

#line 183
    df_0 _S2873 = _S1899;

#line 183
    df_0 _S2874 = df_add_0(&_S2872, &_S2873);

#line 183
    df_0 _S2875 = _S1197;

#line 183
    df_0 _S2876 = _S1146;

#line 183
    df_0 _S2877 = df_add_0(&_S2875, &_S2876);

#line 183
    df_0 _S2878 = _S1206;

#line 183
    df_0 _S2879 = _S1908;

#line 183
    df_0 _S2880 = df_add_0(&_S2878, &_S2879);

#line 183
    df_0 _S2881 = _S1212;

#line 183
    df_0 _S2882 = _S1914;

#line 183
    df_0 _S2883 = df_add_0(&_S2881, &_S2882);

#line 183
    df_0 _S2884 = _S1221;

#line 183
    df_0 _S2885 = _S1923;

#line 183
    df_0 _S2886 = df_add_0(&_S2884, &_S2885);

#line 183
    df_0 _S2887 = _S1230;

#line 183
    df_0 _S2888 = _S1932;

#line 183
    df_0 _S2889 = df_add_0(&_S2887, &_S2888);

#line 183
    df_0 _S2890 = _S1239;

#line 183
    df_0 _S2891 = _S1941;

#line 183
    df_0 _S2892 = df_add_0(&_S2890, &_S2891);

#line 183
    df_0 _S2893 = _S1245;

#line 183
    df_0 _S2894 = _S1947;

#line 183
    df_0 _S2895 = df_add_0(&_S2893, &_S2894);

#line 183
    df_0 _S2896 = _S1254;

#line 183
    df_0 _S2897 = _S1956;

#line 183
    df_0 _S2898 = df_add_0(&_S2896, &_S2897);

#line 183
    df_0 _S2899 = _S1263;

#line 183
    df_0 _S2900 = _S1965;

#line 183
    df_0 _S2901 = df_add_0(&_S2899, &_S2900);

#line 183
    df_0 _S2902 = _S1269;

#line 183
    df_0 _S2903 = _S1245;

#line 183
    df_0 _S2904 = df_add_0(&_S2902, &_S2903);

#line 183
    df_0 _S2905 = _S1278;

#line 183
    df_0 _S2906 = _S1974;

#line 183
    df_0 _S2907 = df_add_0(&_S2905, &_S2906);

#line 183
    df_0 _S2908 = _S1284;

#line 183
    df_0 _S2909 = _S1980;

#line 183
    df_0 _S2910 = df_add_0(&_S2908, &_S2909);

#line 183
    df_0 _S2911 = _S2679;

#line 183
    df_0 _S2912 = _S1986;

#line 183
    df_0 _S2913 = df_add_0(&_S2911, &_S2912);

#line 183
    df_0 _S2914 = _S2682;

#line 183
    df_0 _S2915 = _S2007;

#line 183
    df_0 _S2916 = df_add_0(&_S2914, &_S2915);

#line 183
    df_0 _S2917 = _S2685;

#line 183
    df_0 _S2918 = _S2016;

#line 183
    df_0 _S2919 = df_add_0(&_S2917, &_S2918);

#line 183
    df_0 _S2920 = _S2688;

#line 183
    df_0 _S2921 = _S2025;

#line 183
    df_0 _S2922 = df_add_0(&_S2920, &_S2921);

#line 183
    df_0 _S2923 = _S2691;

#line 183
    df_0 _S2924 = _S2046;

#line 183
    df_0 _S2925 = df_add_0(&_S2923, &_S2924);

#line 183
    df_0 _S2926 = _S2694;

#line 183
    df_0 _S2927 = _S2055;

#line 183
    df_0 _S2928 = df_add_0(&_S2926, &_S2927);

#line 183
    df_0 _S2929 = _S2697;

#line 183
    df_0 _S2930 = _S2064;

#line 183
    df_0 _S2931 = df_add_0(&_S2929, &_S2930);

#line 183
    df_0 _S2932 = _S2700;

#line 183
    df_0 _S2933 = _S2085;

#line 183
    df_0 _S2934 = df_add_0(&_S2932, &_S2933);

#line 183
    df_0 _S2935 = _S2703;

#line 183
    df_0 _S2936 = _S2094;

#line 183
    df_0 _S2937 = df_add_0(&_S2935, &_S2936);

#line 183
    df_0 _S2938 = _S2706;

#line 183
    df_0 _S2939 = _S2103;

#line 183
    df_0 _S2940 = df_add_0(&_S2938, &_S2939);

#line 183
    df_0 _S2941 = _S2709;

#line 183
    df_0 _S2942 = _S2124;

#line 183
    df_0 _S2943 = df_add_0(&_S2941, &_S2942);

#line 183
    df_0 _S2944 = _S2712;

#line 183
    df_0 _S2945 = _S2133;

#line 183
    df_0 _S2946 = df_add_0(&_S2944, &_S2945);

#line 183
    df_0 _S2947 = _S2715;

#line 183
    df_0 _S2948 = _S2139;

#line 183
    df_0 _S2949 = df_add_0(&_S2947, &_S2948);

#line 183
    df_0 _S2950 = _S2718;

#line 183
    df_0 _S2951 = _S2148;

#line 183
    df_0 _S2952 = df_add_0(&_S2950, &_S2951);

#line 183
    df_0 _S2953 = _S2721;

#line 183
    df_0 _S2954 = _S2169;

#line 183
    df_0 _S2955 = df_add_0(&_S2953, &_S2954);

#line 183
    df_0 _S2956 = _S2724;

#line 183
    df_0 _S2957 = _S2178;

#line 183
    df_0 _S2958 = df_add_0(&_S2956, &_S2957);

#line 183
    df_0 _S2959 = _S2727;

#line 183
    df_0 _S2960 = _S2187;

#line 183
    df_0 _S2961 = df_add_0(&_S2959, &_S2960);

#line 183
    df_0 _S2962 = _S2730;

#line 183
    df_0 _S2963 = _S2208;

#line 183
    df_0 _S2964 = df_add_0(&_S2962, &_S2963);

#line 183
    df_0 _S2965 = _S2733;

#line 183
    df_0 _S2966 = _S2217;

#line 183
    df_0 _S2967 = df_add_0(&_S2965, &_S2966);

#line 183
    df_0 _S2968 = _S2736;

#line 183
    df_0 _S2969 = _S2226;

#line 183
    df_0 _S2970 = df_add_0(&_S2968, &_S2969);

#line 183
    df_0 _S2971 = _S2739;

#line 183
    df_0 _S2972 = _S2247;

#line 183
    df_0 _S2973 = df_add_0(&_S2971, &_S2972);

#line 183
    df_0 _S2974 = _S2742;

#line 183
    df_0 _S2975 = _S2256;

#line 183
    df_0 _S2976 = df_add_0(&_S2974, &_S2975);

#line 183
    df_0 _S2977 = _S2745;

#line 183
    df_0 _S2978 = _S2265;

#line 183
    df_0 _S2979 = df_add_0(&_S2977, &_S2978);

#line 183
    df_0 _S2980 = _S2748;

#line 183
    df_0 _S2981 = _S516;

#line 183
    df_0 _S2982 = df_add_0(&_S2980, &_S2981);

#line 183
    df_0 _S2983 = _S2751;

#line 183
    df_0 _S2984 = _S2274;

#line 183
    df_0 _S2985 = df_add_0(&_S2983, &_S2984);

#line 183
    df_0 _S2986 = _S2754;

#line 183
    df_0 _S2987 = _S2283;

#line 183
    df_0 _S2988 = df_add_0(&_S2986, &_S2987);

#line 183
    df_0 _S2989 = _S2757;

#line 183
    df_0 _S2990 = _S543;

#line 183
    df_0 _S2991 = df_add_0(&_S2989, &_S2990);

#line 183
    df_0 _S2992 = _S2760;

#line 183
    df_0 _S2993 = _S2292;

#line 183
    df_0 _S2994 = df_add_0(&_S2992, &_S2993);

#line 183
    df_0 _S2995 = _S2763;

#line 183
    df_0 _S2996 = _S2301;

#line 183
    df_0 _S2997 = df_add_0(&_S2995, &_S2996);

#line 183
    df_0 _S2998 = _S2766;

#line 183
    df_0 _S2999 = _S570;

#line 183
    df_0 _S3000 = df_add_0(&_S2998, &_S2999);

#line 183
    df_0 _S3001 = _S2769;

#line 183
    df_0 _S3002 = _S2310;

#line 183
    df_0 _S3003 = df_add_0(&_S3001, &_S3002);

#line 183
    df_0 _S3004 = _S2772;

#line 183
    df_0 _S3005 = _S2319;

#line 183
    df_0 _S3006 = df_add_0(&_S3004, &_S3005);

#line 183
    df_0 _S3007 = _S2775;

#line 183
    df_0 _S3008 = _S597;

#line 183
    df_0 _S3009 = df_add_0(&_S3007, &_S3008);

#line 183
    df_0 _S3010 = _S2778;

#line 183
    df_0 _S3011 = _S2325;

#line 183
    df_0 _S3012 = df_add_0(&_S3010, &_S3011);

#line 183
    df_0 _S3013 = _S2781;

#line 183
    df_0 _S3014 = _S2334;

#line 183
    df_0 _S3015 = df_add_0(&_S3013, &_S3014);

#line 183
    df_0 _S3016 = _S2784;

#line 183
    df_0 _S3017 = _S2343;

#line 183
    df_0 _S3018 = df_add_0(&_S3016, &_S3017);

#line 183
    df_0 _S3019 = _S2787;

#line 183
    df_0 _S3020 = _S2352;

#line 183
    df_0 _S3021 = df_add_0(&_S3019, &_S3020);

#line 183
    df_0 _S3022 = _S2790;

#line 183
    df_0 _S3023 = _S2367;

#line 183
    df_0 _S3024 = df_add_0(&_S3022, &_S3023);

#line 183
    df_0 _S3025 = _S2793;

#line 183
    df_0 _S3026 = _S2376;

#line 183
    df_0 _S3027 = df_add_0(&_S3025, &_S3026);

#line 183
    df_0 _S3028 = _S2796;

#line 183
    df_0 _S3029 = _S2385;

#line 183
    df_0 _S3030 = df_add_0(&_S3028, &_S3029);

#line 183
    df_0 _S3031 = _S2799;

#line 183
    df_0 _S3032 = _S2400;

#line 183
    df_0 _S3033 = df_add_0(&_S3031, &_S3032);

#line 183
    df_0 _S3034 = _S2802;

#line 183
    df_0 _S3035 = _S2409;

#line 183
    df_0 _S3036 = df_add_0(&_S3034, &_S3035);

#line 183
    df_0 _S3037 = _S2805;

#line 183
    df_0 _S3038 = _S2415;

#line 183
    df_0 _S3039 = df_add_0(&_S3037, &_S3038);

#line 183
    df_0 _S3040 = _S2808;

#line 183
    df_0 _S3041 = _S2424;

#line 183
    df_0 _S3042 = df_add_0(&_S3040, &_S3041);

#line 183
    df_0 _S3043 = _S2811;

#line 183
    df_0 _S3044 = _S2439;

#line 183
    df_0 _S3045 = df_add_0(&_S3043, &_S3044);

#line 183
    df_0 _S3046 = _S2814;

#line 183
    df_0 _S3047 = _S2448;

#line 183
    df_0 _S3048 = df_add_0(&_S3046, &_S3047);

#line 183
    df_0 _S3049 = _S2817;

#line 183
    df_0 _S3050 = _S2457;

#line 183
    df_0 _S3051 = df_add_0(&_S3049, &_S3050);

#line 183
    df_0 _S3052 = _S2820;

#line 183
    df_0 _S3053 = _S2472;

#line 183
    df_0 _S3054 = df_add_0(&_S3052, &_S3053);

#line 183
    df_0 _S3055 = _S2823;

#line 183
    df_0 _S3056 = _S2481;

#line 183
    df_0 _S3057 = df_add_0(&_S3055, &_S3056);

#line 183
    df_0 _S3058 = _S2826;

#line 183
    df_0 _S3059 = _S2490;

#line 183
    df_0 _S3060 = df_add_0(&_S3058, &_S3059);

#line 183
    df_0 _S3061 = _S2829;

#line 183
    df_0 _S3062 = _S888;

#line 183
    df_0 _S3063 = df_add_0(&_S3061, &_S3062);

#line 183
    df_0 _S3064 = _S2832;

#line 183
    df_0 _S3065 = _S2499;

#line 183
    df_0 _S3066 = df_add_0(&_S3064, &_S3065);

#line 183
    df_0 _S3067 = _S2835;

#line 183
    df_0 _S3068 = _S2508;

#line 183
    df_0 _S3069 = df_add_0(&_S3067, &_S3068);

#line 183
    df_0 _S3070 = _S2838;

#line 183
    df_0 _S3071 = _S915;

#line 183
    df_0 _S3072 = df_add_0(&_S3070, &_S3071);

#line 183
    df_0 _S3073 = _S2841;

#line 183
    df_0 _S3074 = _S2517;

#line 183
    df_0 _S3075 = df_add_0(&_S3073, &_S3074);

#line 183
    df_0 _S3076 = _S2844;

#line 183
    df_0 _S3077 = _S2526;

#line 183
    df_0 _S3078 = df_add_0(&_S3076, &_S3077);

#line 183
    df_0 _S3079 = _S2847;

#line 183
    df_0 _S3080 = _S942;

#line 183
    df_0 _S3081 = df_add_0(&_S3079, &_S3080);

#line 183
    df_0 _S3082 = _S2850;

#line 183
    df_0 _S3083 = _S2532;

#line 183
    df_0 _S3084 = df_add_0(&_S3082, &_S3083);

#line 183
    df_0 _S3085 = _S2853;

#line 183
    df_0 _S3086 = _S2541;

#line 183
    df_0 _S3087 = df_add_0(&_S3085, &_S3086);

#line 183
    df_0 _S3088 = _S2856;

#line 183
    df_0 _S3089 = _S2550;

#line 183
    df_0 _S3090 = df_add_0(&_S3088, &_S3089);

#line 183
    df_0 _S3091 = _S2859;

#line 183
    df_0 _S3092 = _S2559;

#line 183
    df_0 _S3093 = df_add_0(&_S3091, &_S3092);

#line 183
    df_0 _S3094 = _S2862;

#line 183
    df_0 _S3095 = _S2568;

#line 183
    df_0 _S3096 = df_add_0(&_S3094, &_S3095);

#line 183
    df_0 _S3097 = _S2865;

#line 183
    df_0 _S3098 = _S2577;

#line 183
    df_0 _S3099 = df_add_0(&_S3097, &_S3098);

#line 183
    df_0 _S3100 = _S2868;

#line 183
    df_0 _S3101 = _S2583;

#line 183
    df_0 _S3102 = df_add_0(&_S3100, &_S3101);

#line 183
    df_0 _S3103 = _S2871;

#line 183
    df_0 _S3104 = _S2592;

#line 183
    df_0 _S3105 = df_add_0(&_S3103, &_S3104);

#line 183
    df_0 _S3106 = _S2874;

#line 183
    df_0 _S3107 = _S2601;

#line 183
    df_0 _S3108 = df_add_0(&_S3106, &_S3107);

#line 183
    df_0 _S3109 = _S2877;

#line 183
    df_0 _S3110 = _S2610;

#line 183
    df_0 _S3111 = df_add_0(&_S3109, &_S3110);

#line 183
    df_0 _S3112 = _S2880;

#line 183
    df_0 _S3113 = _S2619;

#line 183
    df_0 _S3114 = df_add_0(&_S3112, &_S3113);

#line 183
    df_0 _S3115 = _S2883;

#line 183
    df_0 _S3116 = _S1119;

#line 183
    df_0 _S3117 = df_add_0(&_S3115, &_S3116);

#line 183
    df_0 _S3118 = _S2886;

#line 183
    df_0 _S3119 = _S2628;

#line 183
    df_0 _S3120 = df_add_0(&_S3118, &_S3119);

#line 183
    df_0 _S3121 = _S2889;

#line 183
    df_0 _S3122 = _S2637;

#line 183
    df_0 _S3123 = df_add_0(&_S3121, &_S3122);

#line 183
    df_0 _S3124 = _S2892;

#line 183
    df_0 _S3125 = _S1146;

#line 183
    df_0 _S3126 = df_add_0(&_S3124, &_S3125);

#line 183
    df_0 _S3127 = _S2895;

#line 183
    df_0 _S3128 = _S2643;

#line 183
    df_0 _S3129 = df_add_0(&_S3127, &_S3128);

#line 183
    df_0 _S3130 = _S2898;

#line 183
    df_0 _S3131 = _S2652;

#line 183
    df_0 _S3132 = df_add_0(&_S3130, &_S3131);

#line 183
    df_0 _S3133 = _S2901;

#line 183
    df_0 _S3134 = _S2661;

#line 183
    df_0 _S3135 = df_add_0(&_S3133, &_S3134);

#line 183
    df_0 _S3136 = _S2904;

#line 183
    df_0 _S3137 = _S2667;

#line 183
    df_0 _S3138 = df_add_0(&_S3136, &_S3137);

#line 183
    df_0 _S3139 = _S2907;

#line 183
    df_0 _S3140 = _S2676;

#line 183
    df_0 _S3141 = df_add_0(&_S3139, &_S3140);

#line 183
    df_0 _S3142 = _S2910;

#line 183
    df_0 _S3143 = _S1245;

#line 183
    df_0 _S3144 = df_add_0(&_S3142, &_S3143);

#line 1575
    float sg_0 = (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->sign_0;
    df_0 * _S3145 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0]);

#line 1576
    df_0 _S3146 = _S2913;

#line 1576
    df_0 _S3147 = df_mulf_0(&_S3146, sg_0);

#line 1576
    *_S3145 = _S3147;
    df_0 * _S3148 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 1U]);

#line 1577
    df_0 _S3149 = _S2916;

#line 1577
    df_0 _S3150 = df_mulf_0(&_S3149, sg_0);

#line 1577
    *_S3148 = _S3150;
    df_0 * _S3151 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 2U]);

#line 1578
    df_0 _S3152 = _S2919;

#line 1578
    df_0 _S3153 = df_mulf_0(&_S3152, sg_0);

#line 1578
    *_S3151 = _S3153;
    df_0 * _S3154 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 3U]);

#line 1579
    df_0 _S3155 = _S2922;

#line 1579
    df_0 _S3156 = df_mulf_0(&_S3155, sg_0);

#line 1579
    *_S3154 = _S3156;
    df_0 * _S3157 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 4U]);

#line 1580
    df_0 _S3158 = _S2925;

#line 1580
    df_0 _S3159 = df_mulf_0(&_S3158, sg_0);

#line 1580
    *_S3157 = _S3159;
    df_0 * _S3160 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 5U]);

#line 1581
    df_0 _S3161 = _S2928;

#line 1581
    df_0 _S3162 = df_mulf_0(&_S3161, sg_0);

#line 1581
    *_S3160 = _S3162;
    df_0 * _S3163 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 6U]);

#line 1582
    df_0 _S3164 = _S2931;

#line 1582
    df_0 _S3165 = df_mulf_0(&_S3164, sg_0);

#line 1582
    *_S3163 = _S3165;
    df_0 * _S3166 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 7U]);

#line 1583
    df_0 _S3167 = _S2934;

#line 1583
    df_0 _S3168 = df_mulf_0(&_S3167, sg_0);

#line 1583
    *_S3166 = _S3168;
    df_0 * _S3169 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 8U]);

#line 1584
    df_0 _S3170 = _S2937;

#line 1584
    df_0 _S3171 = df_mulf_0(&_S3170, sg_0);

#line 1584
    *_S3169 = _S3171;
    df_0 * _S3172 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 9U]);

#line 1585
    df_0 _S3173 = _S2940;

#line 1585
    df_0 _S3174 = df_mulf_0(&_S3173, sg_0);

#line 1585
    *_S3172 = _S3174;
    df_0 * _S3175 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 10U]);

#line 1586
    df_0 _S3176 = _S2943;

#line 1586
    df_0 _S3177 = df_mulf_0(&_S3176, sg_0);

#line 1586
    *_S3175 = _S3177;
    df_0 * _S3178 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 11U]);

#line 1587
    df_0 _S3179 = _S2946;

#line 1587
    df_0 _S3180 = df_mulf_0(&_S3179, sg_0);

#line 1587
    *_S3178 = _S3180;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 12U]) = _S3150;
    df_0 * _S3181 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 13U]);

#line 1589
    df_0 _S3182 = _S2949;

#line 1589
    df_0 _S3183 = df_mulf_0(&_S3182, sg_0);

#line 1589
    *_S3181 = _S3183;
    df_0 * _S3184 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 14U]);

#line 1590
    df_0 _S3185 = _S2952;

#line 1590
    df_0 _S3186 = df_mulf_0(&_S3185, sg_0);

#line 1590
    *_S3184 = _S3186;
    df_0 * _S3187 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 15U]);

#line 1591
    df_0 _S3188 = _S2955;

#line 1591
    df_0 _S3189 = df_mulf_0(&_S3188, sg_0);

#line 1591
    *_S3187 = _S3189;
    df_0 * _S3190 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 16U]);

#line 1592
    df_0 _S3191 = _S2958;

#line 1592
    df_0 _S3192 = df_mulf_0(&_S3191, sg_0);

#line 1592
    *_S3190 = _S3192;
    df_0 * _S3193 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 17U]);

#line 1593
    df_0 _S3194 = _S2961;

#line 1593
    df_0 _S3195 = df_mulf_0(&_S3194, sg_0);

#line 1593
    *_S3193 = _S3195;
    df_0 * _S3196 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 18U]);

#line 1594
    df_0 _S3197 = _S2964;

#line 1594
    df_0 _S3198 = df_mulf_0(&_S3197, sg_0);

#line 1594
    *_S3196 = _S3198;
    df_0 * _S3199 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 19U]);

#line 1595
    df_0 _S3200 = _S2967;

#line 1595
    df_0 _S3201 = df_mulf_0(&_S3200, sg_0);

#line 1595
    *_S3199 = _S3201;
    df_0 * _S3202 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 20U]);

#line 1596
    df_0 _S3203 = _S2970;

#line 1596
    df_0 _S3204 = df_mulf_0(&_S3203, sg_0);

#line 1596
    *_S3202 = _S3204;
    df_0 * _S3205 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 21U]);

#line 1597
    df_0 _S3206 = _S2973;

#line 1597
    df_0 _S3207 = df_mulf_0(&_S3206, sg_0);

#line 1597
    *_S3205 = _S3207;
    df_0 * _S3208 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 22U]);

#line 1598
    df_0 _S3209 = _S2976;

#line 1598
    df_0 _S3210 = df_mulf_0(&_S3209, sg_0);

#line 1598
    *_S3208 = _S3210;
    df_0 * _S3211 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 23U]);

#line 1599
    df_0 _S3212 = _S2979;

#line 1599
    df_0 _S3213 = df_mulf_0(&_S3212, sg_0);

#line 1599
    *_S3211 = _S3213;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 24U]) = _S3153;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 25U]) = _S3186;
    df_0 * _S3214 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 26U]);

#line 1602
    df_0 _S3215 = _S2982;

#line 1602
    df_0 _S3216 = df_mulf_0(&_S3215, sg_0);

#line 1602
    *_S3214 = _S3216;
    df_0 * _S3217 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 27U]);

#line 1603
    df_0 _S3218 = _S2985;

#line 1603
    df_0 _S3219 = df_mulf_0(&_S3218, sg_0);

#line 1603
    *_S3217 = _S3219;
    df_0 * _S3220 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 28U]);

#line 1604
    df_0 _S3221 = _S2988;

#line 1604
    df_0 _S3222 = df_mulf_0(&_S3221, sg_0);

#line 1604
    *_S3220 = _S3222;
    df_0 * _S3223 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 29U]);

#line 1605
    df_0 _S3224 = _S2991;

#line 1605
    df_0 _S3225 = df_mulf_0(&_S3224, sg_0);

#line 1605
    *_S3223 = _S3225;
    df_0 * _S3226 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 30U]);

#line 1606
    df_0 _S3227 = _S2994;

#line 1606
    df_0 _S3228 = df_mulf_0(&_S3227, sg_0);

#line 1606
    *_S3226 = _S3228;
    df_0 * _S3229 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 31U]);

#line 1607
    df_0 _S3230 = _S2997;

#line 1607
    df_0 _S3231 = df_mulf_0(&_S3230, sg_0);

#line 1607
    *_S3229 = _S3231;
    df_0 * _S3232 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 32U]);

#line 1608
    df_0 _S3233 = _S3000;

#line 1608
    df_0 _S3234 = df_mulf_0(&_S3233, sg_0);

#line 1608
    *_S3232 = _S3234;
    df_0 * _S3235 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 33U]);

#line 1609
    df_0 _S3236 = _S3003;

#line 1609
    df_0 _S3237 = df_mulf_0(&_S3236, sg_0);

#line 1609
    *_S3235 = _S3237;
    df_0 * _S3238 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 34U]);

#line 1610
    df_0 _S3239 = _S3006;

#line 1610
    df_0 _S3240 = df_mulf_0(&_S3239, sg_0);

#line 1610
    *_S3238 = _S3240;
    df_0 * _S3241 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 35U]);

#line 1611
    df_0 _S3242 = _S3009;

#line 1611
    df_0 _S3243 = df_mulf_0(&_S3242, sg_0);

#line 1611
    *_S3241 = _S3243;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 36U]) = _S3156;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 37U]) = _S3189;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 38U]) = _S3219;
    df_0 * _S3244 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 39U]);

#line 1615
    df_0 _S3245 = _S3012;

#line 1615
    df_0 _S3246 = df_mulf_0(&_S3245, sg_0);

#line 1615
    *_S3244 = _S3246;
    df_0 * _S3247 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 40U]);

#line 1616
    df_0 _S3248 = _S3015;

#line 1616
    df_0 _S3249 = df_mulf_0(&_S3248, sg_0);

#line 1616
    *_S3247 = _S3249;
    df_0 * _S3250 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 41U]);

#line 1617
    df_0 _S3251 = _S3018;

#line 1617
    df_0 _S3252 = df_mulf_0(&_S3251, sg_0);

#line 1617
    *_S3250 = _S3252;
    df_0 * _S3253 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 42U]);

#line 1618
    df_0 _S3254 = _S3021;

#line 1618
    df_0 _S3255 = df_mulf_0(&_S3254, sg_0);

#line 1618
    *_S3253 = _S3255;
    df_0 * _S3256 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 43U]);

#line 1619
    df_0 _S3257 = _S3024;

#line 1619
    df_0 _S3258 = df_mulf_0(&_S3257, sg_0);

#line 1619
    *_S3256 = _S3258;
    df_0 * _S3259 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 44U]);

#line 1620
    df_0 _S3260 = _S3027;

#line 1620
    df_0 _S3261 = df_mulf_0(&_S3260, sg_0);

#line 1620
    *_S3259 = _S3261;
    df_0 * _S3262 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 45U]);

#line 1621
    df_0 _S3263 = _S3030;

#line 1621
    df_0 _S3264 = df_mulf_0(&_S3263, sg_0);

#line 1621
    *_S3262 = _S3264;
    df_0 * _S3265 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 46U]);

#line 1622
    df_0 _S3266 = _S3033;

#line 1622
    df_0 _S3267 = df_mulf_0(&_S3266, sg_0);

#line 1622
    *_S3265 = _S3267;
    df_0 * _S3268 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 47U]);

#line 1623
    df_0 _S3269 = _S3036;

#line 1623
    df_0 _S3270 = df_mulf_0(&_S3269, sg_0);

#line 1623
    *_S3268 = _S3270;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 48U]) = _S3159;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 49U]) = _S3192;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 50U]) = _S3222;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 51U]) = _S3249;
    df_0 * _S3271 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 52U]);

#line 1628
    df_0 _S3272 = _S3039;

#line 1628
    df_0 _S3273 = df_mulf_0(&_S3272, sg_0);

#line 1628
    *_S3271 = _S3273;
    df_0 * _S3274 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 53U]);

#line 1629
    df_0 _S3275 = _S3042;

#line 1629
    df_0 _S3276 = df_mulf_0(&_S3275, sg_0);

#line 1629
    *_S3274 = _S3276;
    df_0 * _S3277 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 54U]);

#line 1630
    df_0 _S3278 = _S3045;

#line 1630
    df_0 _S3279 = df_mulf_0(&_S3278, sg_0);

#line 1630
    *_S3277 = _S3279;
    df_0 * _S3280 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 55U]);

#line 1631
    df_0 _S3281 = _S3048;

#line 1631
    df_0 _S3282 = df_mulf_0(&_S3281, sg_0);

#line 1631
    *_S3280 = _S3282;
    df_0 * _S3283 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 56U]);

#line 1632
    df_0 _S3284 = _S3051;

#line 1632
    df_0 _S3285 = df_mulf_0(&_S3284, sg_0);

#line 1632
    *_S3283 = _S3285;
    df_0 * _S3286 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 57U]);

#line 1633
    df_0 _S3287 = _S3054;

#line 1633
    df_0 _S3288 = df_mulf_0(&_S3287, sg_0);

#line 1633
    *_S3286 = _S3288;
    df_0 * _S3289 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 58U]);

#line 1634
    df_0 _S3290 = _S3057;

#line 1634
    df_0 _S3291 = df_mulf_0(&_S3290, sg_0);

#line 1634
    *_S3289 = _S3291;
    df_0 * _S3292 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 59U]);

#line 1635
    df_0 _S3293 = _S3060;

#line 1635
    df_0 _S3294 = df_mulf_0(&_S3293, sg_0);

#line 1635
    *_S3292 = _S3294;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 60U]) = _S3162;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 61U]) = _S3195;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 62U]) = _S3225;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 63U]) = _S3252;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 64U]) = _S3276;
    df_0 * _S3295 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 65U]);

#line 1641
    df_0 _S3296 = _S3063;

#line 1641
    df_0 _S3297 = df_mulf_0(&_S3296, sg_0);

#line 1641
    *_S3295 = _S3297;
    df_0 * _S3298 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 66U]);

#line 1642
    df_0 _S3299 = _S3066;

#line 1642
    df_0 _S3300 = df_mulf_0(&_S3299, sg_0);

#line 1642
    *_S3298 = _S3300;
    df_0 * _S3301 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 67U]);

#line 1643
    df_0 _S3302 = _S3069;

#line 1643
    df_0 _S3303 = df_mulf_0(&_S3302, sg_0);

#line 1643
    *_S3301 = _S3303;
    df_0 * _S3304 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 68U]);

#line 1644
    df_0 _S3305 = _S3072;

#line 1644
    df_0 _S3306 = df_mulf_0(&_S3305, sg_0);

#line 1644
    *_S3304 = _S3306;
    df_0 * _S3307 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 69U]);

#line 1645
    df_0 _S3308 = _S3075;

#line 1645
    df_0 _S3309 = df_mulf_0(&_S3308, sg_0);

#line 1645
    *_S3307 = _S3309;
    df_0 * _S3310 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 70U]);

#line 1646
    df_0 _S3311 = _S3078;

#line 1646
    df_0 _S3312 = df_mulf_0(&_S3311, sg_0);

#line 1646
    *_S3310 = _S3312;
    df_0 * _S3313 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 71U]);

#line 1647
    df_0 _S3314 = _S3081;

#line 1647
    df_0 _S3315 = df_mulf_0(&_S3314, sg_0);

#line 1647
    *_S3313 = _S3315;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 72U]) = _S3165;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 73U]) = _S3198;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 74U]) = _S3228;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 75U]) = _S3255;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 76U]) = _S3279;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 77U]) = _S3300;
    df_0 * _S3316 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 78U]);

#line 1654
    df_0 _S3317 = _S3084;

#line 1654
    df_0 _S3318 = df_mulf_0(&_S3317, sg_0);

#line 1654
    *_S3316 = _S3318;
    df_0 * _S3319 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 79U]);

#line 1655
    df_0 _S3320 = _S3087;

#line 1655
    df_0 _S3321 = df_mulf_0(&_S3320, sg_0);

#line 1655
    *_S3319 = _S3321;
    df_0 * _S3322 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 80U]);

#line 1656
    df_0 _S3323 = _S3090;

#line 1656
    df_0 _S3324 = df_mulf_0(&_S3323, sg_0);

#line 1656
    *_S3322 = _S3324;
    df_0 * _S3325 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 81U]);

#line 1657
    df_0 _S3326 = _S3093;

#line 1657
    df_0 _S3327 = df_mulf_0(&_S3326, sg_0);

#line 1657
    *_S3325 = _S3327;
    df_0 * _S3328 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 82U]);

#line 1658
    df_0 _S3329 = _S3096;

#line 1658
    df_0 _S3330 = df_mulf_0(&_S3329, sg_0);

#line 1658
    *_S3328 = _S3330;
    df_0 * _S3331 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 83U]);

#line 1659
    df_0 _S3332 = _S3099;

#line 1659
    df_0 _S3333 = df_mulf_0(&_S3332, sg_0);

#line 1659
    *_S3331 = _S3333;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 84U]) = _S3168;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 85U]) = _S3201;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 86U]) = _S3231;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 87U]) = _S3258;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 88U]) = _S3282;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 89U]) = _S3303;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 90U]) = _S3321;
    df_0 * _S3334 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 91U]);

#line 1667
    df_0 _S3335 = _S3102;

#line 1667
    df_0 _S3336 = df_mulf_0(&_S3335, sg_0);

#line 1667
    *_S3334 = _S3336;
    df_0 * _S3337 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 92U]);

#line 1668
    df_0 _S3338 = _S3105;

#line 1668
    df_0 _S3339 = df_mulf_0(&_S3338, sg_0);

#line 1668
    *_S3337 = _S3339;
    df_0 * _S3340 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 93U]);

#line 1669
    df_0 _S3341 = _S3108;

#line 1669
    df_0 _S3342 = df_mulf_0(&_S3341, sg_0);

#line 1669
    *_S3340 = _S3342;
    df_0 * _S3343 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 94U]);

#line 1670
    df_0 _S3344 = _S3111;

#line 1670
    df_0 _S3345 = df_mulf_0(&_S3344, sg_0);

#line 1670
    *_S3343 = _S3345;
    df_0 * _S3346 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 95U]);

#line 1671
    df_0 _S3347 = _S3114;

#line 1671
    df_0 _S3348 = df_mulf_0(&_S3347, sg_0);

#line 1671
    *_S3346 = _S3348;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 96U]) = _S3171;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 97U]) = _S3204;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 98U]) = _S3234;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 99U]) = _S3261;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 100U]) = _S3285;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 101U]) = _S3306;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 102U]) = _S3324;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 103U]) = _S3339;
    df_0 * _S3349 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 104U]);

#line 1680
    df_0 _S3350 = _S3117;

#line 1680
    df_0 _S3351 = df_mulf_0(&_S3350, sg_0);

#line 1680
    *_S3349 = _S3351;
    df_0 * _S3352 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 105U]);

#line 1681
    df_0 _S3353 = _S3120;

#line 1681
    df_0 _S3354 = df_mulf_0(&_S3353, sg_0);

#line 1681
    *_S3352 = _S3354;
    df_0 * _S3355 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 106U]);

#line 1682
    df_0 _S3356 = _S3123;

#line 1682
    df_0 _S3357 = df_mulf_0(&_S3356, sg_0);

#line 1682
    *_S3355 = _S3357;
    df_0 * _S3358 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 107U]);

#line 1683
    df_0 _S3359 = _S3126;

#line 1683
    df_0 _S3360 = df_mulf_0(&_S3359, sg_0);

#line 1683
    *_S3358 = _S3360;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 108U]) = _S3174;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 109U]) = _S3207;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 110U]) = _S3237;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 111U]) = _S3264;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 112U]) = _S3288;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 113U]) = _S3309;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 114U]) = _S3327;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 115U]) = _S3342;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 116U]) = _S3354;
    df_0 * _S3361 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 117U]);

#line 1693
    df_0 _S3362 = _S3129;

#line 1693
    df_0 _S3363 = df_mulf_0(&_S3362, sg_0);

#line 1693
    *_S3361 = _S3363;
    df_0 * _S3364 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 118U]);

#line 1694
    df_0 _S3365 = _S3132;

#line 1694
    df_0 _S3366 = df_mulf_0(&_S3365, sg_0);

#line 1694
    *_S3364 = _S3366;
    df_0 * _S3367 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 119U]);

#line 1695
    df_0 _S3368 = _S3135;

#line 1695
    df_0 _S3369 = df_mulf_0(&_S3368, sg_0);

#line 1695
    *_S3367 = _S3369;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 120U]) = _S3177;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 121U]) = _S3210;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 122U]) = _S3240;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 123U]) = _S3267;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 124U]) = _S3291;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 125U]) = _S3312;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 126U]) = _S3330;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 127U]) = _S3345;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 128U]) = _S3357;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 129U]) = _S3366;
    df_0 * _S3370 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 130U]);

#line 1706
    df_0 _S3371 = _S3138;

#line 1706
    df_0 _S3372 = df_mulf_0(&_S3371, sg_0);

#line 1706
    *_S3370 = _S3372;
    df_0 * _S3373 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 131U]);

#line 1707
    df_0 _S3374 = _S3141;

#line 1707
    df_0 _S3375 = df_mulf_0(&_S3374, sg_0);

#line 1707
    *_S3373 = _S3375;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 132U]) = _S3180;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 133U]) = _S3213;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 134U]) = _S3243;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 135U]) = _S3270;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 136U]) = _S3294;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 137U]) = _S3315;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 138U]) = _S3333;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 139U]) = _S3348;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 140U]) = _S3360;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 141U]) = _S3369;
    *(&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 142U]) = _S3375;
    df_0 * _S3376 = (&((&kernelContext_0)->globalParams_0->blocks_0)[ob_0 + 143U]);

#line 1719
    df_0 _S3377 = _S3144;

#line 1719
    df_0 _S3378 = df_mulf_0(&_S3377, sg_0);

#line 1719
    *_S3376 = _S3378;
    return;
}

// [numthreads(64, 1, 1)]
SLANG_PRELUDE_EXPORT
void main_0_Thread(ComputeThreadVaryingInput* varyingInput, void* entryPointParams, void* globalParams)
{
    _main_0(varyingInput, entryPointParams, globalParams);
}
// [numthreads(64, 1, 1)]
SLANG_PRELUDE_EXPORT
void main_0_Group(ComputeVaryingInput* varyingInput, void* entryPointParams, void* globalParams)
{
    ComputeThreadVaryingInput threadInput = {};
    threadInput.groupID = varyingInput->startGroupID;
    for (uint32_t x = 0; x < 64; ++x)
    {
        threadInput.groupThreadID.x = x;
        _main_0(&threadInput, entryPointParams, globalParams);
    }
}
// [numthreads(64, 1, 1)]
SLANG_PRELUDE_EXPORT
void main_0(ComputeVaryingInput* varyingInput, void* entryPointParams, void* globalParams)
{
    ComputeVaryingInput vi = *varyingInput;
    ComputeVaryingInput groupVaryingInput = {};
    for (uint32_t z = vi.startGroupID.z; z < vi.endGroupID.z; ++z)
    {
        groupVaryingInput.startGroupID.z = z;
        for (uint32_t y = vi.startGroupID.y; y < vi.endGroupID.y; ++y)
        {
            groupVaryingInput.startGroupID.y = y;
            for (uint32_t x = vi.startGroupID.x; x < vi.endGroupID.x; ++x)
            {
                groupVaryingInput.startGroupID.x = x;
                main_0_Group(&groupVaryingInput, entryPointParams, globalParams);
            }
        }
    }
}
