module {
  aie.device(npu1_1col) @xclbin_device {
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
      aiex.npu.write32 {address = 2224160 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224176 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224128 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224144 : ui32, value = 0 : ui32}
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
      aiex.npu.write32 {address = 258100 : ui32, value = 2147483653 : ui32}
      aiex.npu.write32 {address = 258324 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 258064 : ui32, value = 2147483663 : ui32}
      aiex.npu.write32 {address = 258364 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 258056 : ui32, value = 3221225541 : ui32}
      aiex.npu.write32 {address = 258304 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 258560 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1769520 : ui32, value = 2147483656 : ui32}
      aiex.npu.write32 {address = 1769760 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1769504 : ui32, value = 2147483662 : ui32}
      aiex.npu.write32 {address = 1769784 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2355204 : ui32, value = 2147483654 : ui32}
      aiex.npu.write32 {address = 2355480 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2355224 : ui32, value = 2147483649 : ui32}
      aiex.npu.write32 {address = 2355460 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 126976 : ui32, mask = 3072 : ui32, value = 1024 : ui32}
      aiex.npu.maskwrite32 {address = 126980 : ui32, mask = 48 : ui32, value = 16 : ui32}
      aiex.npu.maskwrite32 {address = 2301952 : ui32, mask = 1 : ui32, value = 1 : ui32}
      aie.end
    }
  }
}
