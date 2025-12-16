#pragma once

/*
 * GSPLAT_FOR_EACH(macro, a, b, c, ...):
 *   Expands to: macro(a) macro(b) macro(c) ...
 *   - Works with zero or more arguments.
 *   - When called with no arguments, it expands to nothing.
 *   - Example:
 *       #define F(x) FEATURE_ITEM(x)
 *       GSPLAT_FOR_EACH(F, 1, 2, 3)
 *     expands to:
 *       FEATURE_ITEM(1) FEATURE_ITEM(2) FEATURE_ITEM(3)
 */

#define GSPLAT_PARENS ()

#define GSPLAT_EXPAND(...) GSPLAT_EXPAND4(GSPLAT_EXPAND4(GSPLAT_EXPAND4(GSPLAT_EXPAND4(__VA_ARGS__))))
#define GSPLAT_EXPAND4(...) GSPLAT_EXPAND3(GSPLAT_EXPAND3(GSPLAT_EXPAND3(GSPLAT_EXPAND3(__VA_ARGS__))))
#define GSPLAT_EXPAND3(...) GSPLAT_EXPAND2(GSPLAT_EXPAND2(GSPLAT_EXPAND2(GSPLAT_EXPAND2(__VA_ARGS__))))
#define GSPLAT_EXPAND2(...) GSPLAT_EXPAND1(GSPLAT_EXPAND1(GSPLAT_EXPAND1(GSPLAT_EXPAND1(__VA_ARGS__))))
#define GSPLAT_EXPAND1(...) __VA_ARGS__

#define GSPLAT_FOR_EACH(macro, ...)                                    \
  __VA_OPT__(GSPLAT_EXPAND(GSPLAT_FOR_EACH_HELPER(macro, __VA_ARGS__)))
#define GSPLAT_FOR_EACH_HELPER(macro, a1, ...)                         \
  macro(a1)                                                     \
  __VA_OPT__(GSPLAT_FOR_EACH_AGAIN GSPLAT_PARENS (macro, __VA_ARGS__))
#define GSPLAT_FOR_EACH_AGAIN() GSPLAT_FOR_EACH_HELPER
