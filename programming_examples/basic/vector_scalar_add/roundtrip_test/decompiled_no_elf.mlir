module {
  aie.device(npu1_1col) @xclbin_device {
    %tile_0_2 = aie.tile(0, 2)
    aie.runtime_sequence @configure() {
      aiex.npu.maskwrite32 {address = 2301952 : ui32, mask = 1 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 2219536 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 2219544 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 2219520 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 2219528 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.write32 {address = 2228224 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 2219536 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 2219544 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 2219520 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 2219528 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 2301952 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 2301952 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224128 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224144 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224160 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224176 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224192 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224208 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224224 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224240 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224256 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224272 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224288 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224304 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224320 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224336 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224352 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224368 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 81952 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 81968 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1835040 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1835056 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224160 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224176 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224128 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224144 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1835008 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1835024 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 81920 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 81936 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2215936 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2216000 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2216032 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2215968 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2219524 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 2219520 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 2219540 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 2219536 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 1703936 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1704000 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1704704 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1704768 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1704800 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1704736 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1704032 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1703968 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1705476 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 1705472 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 1705524 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 1705520 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 1705484 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 1705480 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 1705532 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 1705528 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 258112 : ui32, value = 2147483653 : ui32}
      aiex.npu.write32 {address = 258324 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 258064 : ui32, value = 2147483664 : ui32}
      aiex.npu.write32 {address = 258368 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 258056 : ui32, value = 3221225541 : ui32}
      aiex.npu.write32 {address = 258304 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 258560 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1769472 : ui32, value = 2147483659 : ui32}
      aiex.npu.write32 {address = 1769772 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1769520 : ui32, value = 2147483648 : ui32}
      aiex.npu.write32 {address = 1769728 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1769476 : ui32, value = 2147483661 : ui32}
      aiex.npu.write32 {address = 1769780 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1769508 : ui32, value = 2147483649 : ui32}
      aiex.npu.write32 {address = 1769732 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2355204 : ui32, value = 2147483654 : ui32}
      aiex.npu.write32 {address = 2355480 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2355220 : ui32, value = 2147483649 : ui32}
      aiex.npu.write32 {address = 2355460 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 126976 : ui32, mask = 3072 : ui32, value = 1024 : ui32}
      aiex.npu.maskwrite32 {address = 126980 : ui32, mask = 48 : ui32, value = 16 : ui32}
      aiex.npu.maskwrite32 {address = 2301952 : ui32, mask = 1 : ui32, value = 1 : ui32}
      aie.end
    }
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } 
  }
}
