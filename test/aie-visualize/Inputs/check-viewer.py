# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Drives an aie-visualize viewer in a headless browser and prints what a
reader would see.

Usage: check-viewer.py view.html
       check-viewer.py aie-visualize.html first.svg second.svg ...

The second form opens aie-visualize.html?svg=first.svg&svg=second.svg, with
the files served from where they are."""

import pathlib
import sys

from playwright.sync_api import sync_playwright


def main():
    html = pathlib.Path(sys.argv[1]).resolve()
    svgs = [pathlib.Path(path).resolve() for path in sys.argv[2:]]
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.on("pageerror", lambda e: errors.append(str(e)))
        if svgs:
            served = {"/viewer.html": html, **{f"/{svg.name}": svg for svg in svgs}}
            page.route(
                "http://viz.test/**",
                lambda route: route.fulfill(
                    path=str(
                        served[route.request.url.split("viz.test")[1].split("?")[0]]
                    )
                ),
            )
            query = "&".join(f"svg={svg.name}" for svg in svgs)
            page.goto(f"http://viz.test/viewer.html?{query}")
        else:
            page.goto(html.as_uri())
        page.wait_for_timeout(300)
        count = lambda selector: page.locator(selector).count()
        active = ".view.active "

        print("views:", " ".join(page.locator("#tabs button").all_inner_texts()))
        svg = page.locator(active + "svg")
        print(
            "dimmed at rest:",
            str(svg.evaluate("s => s.classList.contains('focus')")).lower(),
        )

        page.uncheck("#downstream")
        page.locator(active + '[data-id="arg:0"]').click()
        print("upstream of %arg0: edges", count(active + ".edge.hl"))
        page.check("#downstream")
        page.uncheck("#upstream")
        print("downstream of %arg0: calls", count(active + ".node.call.hl"))
        page.check("#upstream")
        page.uncheck("#downstream")
        page.locator(active + '[data-id="arg:1"]').click()
        print("upstream of %arg1: calls", count(active + ".node.call.hl"))
        page.check("#downstream")

        page.locator(active + '[data-id="call:0,2#0"]').click()
        hit = page.locator("#code .line.hit .n").all_inner_texts()
        print("kernel call sources:", "broadcast.mlir:" + ",".join(hit))

        page.locator("#tabs button", has_text="Physical").click()
        page.locator(active + '[data-id="dma:0,2:S2MM0"]').click()
        page.locator("#files button", has_text="as read").click()
        print("DMA channel program lines:", count("#code .line.hit"))

        widths = page.locator(active + "path.line").evaluate_all(
            "ps => [...new Set(ps.map(p => p.getAttribute('stroke-width')))]"
        )
        print("widths at rest:", " ".join(widths))
        browser.close()
    print("page errors:", len(errors))
    for error in errors:
        print(error)


if __name__ == "__main__":
    main()
