// Copyright (c) 2020 stillwwater
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would be
//    appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

#ifndef MATHF_CONFIG_H
#define MATHF_CONFIG_H

// SSE instrinsics
#if defined(__SSE__)
#    define MATHF_SSE
#elif defined(_MSC_VER)
#    if defined(_M_AMD64) || defined(_M_X64)
#        define MATHF_SSE
#    elif _M_IX86_FP
#        define MATHF_SSE
#    endif
#endif

#endif // MATHF_CONFIG_H
