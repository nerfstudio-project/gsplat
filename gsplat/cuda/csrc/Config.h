#pragma once

// If user wants to explicitly build at least one, they have to specify all modules they want to build
// If they don't specify any, or they don't want to build some, all the others that aren't specified
// will be built.
// Ex:
//   GSPLAT_BUILD_3DGUT=1 GSPLAT_BUILD_2DGS=1 -> build 3dgut and 2dgs only, do not build anything else.
//   GSPLAT_BUILD_3DGUT=0 GSPLAT_BUILD_2DGS=0 -> build everything except 3dgut and 2dgs
//   <no GSPLAT_BUILD_* defined> -> build everything

#if defined(GSPLAT_BUILD_2DGS) && GSPLAT_BUILD_2DGS || \
    defined(GSPLAT_BUILD_3DGS) && GSPLAT_BUILD_3DGS || \
    defined(GSPLAT_BUILD_3DGUT) && GSPLAT_BUILD_3DGUT || \
    defined(GSPLAT_BUILD_ADAM) && GSPLAT_BUILD_ADAM || \
    defined(GSPLAT_BUILD_RELOC) && GSPLAT_BUILD_RELOC

#   define GSPLAT_DEFAULT_ENABLE_BUILD 0
#else
#   define GSPLAT_DEFAULT_ENABLE_BUILD 1
#endif

#ifndef GSPLAT_BUILD_2DGS
#   define GSPLAT_BUILD_2DGS GSPLAT_DEFAULT_ENABLE_BUILD
#endif

#ifndef GSPLAT_BUILD_3DGS
#   define GSPLAT_BUILD_3DGS GSPLAT_DEFAULT_ENABLE_BUILD
#endif

#ifndef GSPLAT_BUILD_3DGUT
#   define GSPLAT_BUILD_3DGUT GSPLAT_DEFAULT_ENABLE_BUILD
#endif

#ifndef GSPLAT_BUILD_ADAM
#   define GSPLAT_BUILD_ADAM GSPLAT_DEFAULT_ENABLE_BUILD
#endif

#ifndef GSPLAT_BUILD_RELOC
#   define GSPLAT_BUILD_RELOC GSPLAT_DEFAULT_ENABLE_BUILD
#endif

#ifndef GSPLAT_NUM_CHANNELS
#   define GSPLAT_NUM_CHANNELS 1,2,3,4,5,8,9,16,17,32,33,64,65,128,129,256,257,512,513
#endif
