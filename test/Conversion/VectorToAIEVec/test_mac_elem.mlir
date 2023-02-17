// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aieml" | FileCheck %s

func.func @test_mac_elem(%a : vector<16xi32>, %b : vector<16xi32>, %c : vector<16xi32>) -> vector<16xi32> {
    %0 = vector.fma %a, %b, %c : vector<16xi32>
    return %0 : vector<16xi32>
}

// CHECK-LABEL: func @test_mac_elem
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xi32>
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<16xi32>
// CHECK-SAME: %[[C:[A-Za-z0-9]+]]: vector<16xi32>
//      CHECK: %[[UPS:.*]] = aievec.ups %[[C]] {shift = 0 : i8} : vector<16xi32>, vector<16xi64>
//      CHECK: %[[ME:.*]] = aievec.mac_elem %[[A]], %[[B]], %[[UPS:.*]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
//      CHECK: %[[RES:.*]] = aievec.srs %[[ME:.*]] {shift = 0 : i8} : vector<16xi64>, vector<16xi32>
//      CHECK: return %[[RES:.*]] : vector<16xi32>