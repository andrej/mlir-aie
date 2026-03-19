// RuntimeSequenceOp
//===----------------------------------------------------------------------===//

ParseResult AIEX::RuntimeSequenceOp::parse(OpAsmParser &parser,
                                           OperationState &result) {

  // Name of this runtime sequence
  StringAttr nameAttr;
  (void)parser.parseOptionalSymbolName(
      nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes);

  SmallVector<OpAsmParser::Argument> entryArgs;

  // Entry arguments,  e.g. (%addr: memref<1xi32>)
  ParseResult argParseResult = parser.parseCommaSeparatedList(
      OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
        OpAsmParser::Argument argument;
        if (parser.parseArgument(argument, true, true)) {
          return failure();
        }
        entryArgs.push_back(argument);
        return success();
      });
  if (argParseResult) {
    return argParseResult;
  }

  // Body
  auto *body = result.addRegion();
  ParseResult bodyParseResult = parser.parseRegion(*body, entryArgs, false);
  if (bodyParseResult) {
    return bodyParseResult;
  }

  return success();
}

void AIEX::RuntimeSequenceOp::print(OpAsmPrinter &printer) {
  Region &body = getRegion();

  auto nameAttr = (*this)->getAttrOfType<StringAttr>(
      mlir::SymbolTable::getSymbolAttrName());
  if (nameAttr &&
      nameAttr != ::mlir::OpBuilder((*this)->getContext())
                      .getStringAttr(getDefaultRuntimeSequenceName())) {
    printer << ' ';
    printer.printSymbolName(nameAttr);
  }

  printer << '(';
  for (unsigned i = 0, n = body.getNumArguments(); i < n; i++) {
    if (i > 0) {
      printer << ", ";
    }
    printer.printRegionArgument(body.getArgument(i));
  }
  printer << ')';

  printer << ' ';
  printer.printRegion(body, false, true);
}

LogicalResult AIEX::RuntimeSequenceOp::verify() {
  AIE::DeviceOp device = (*this)->getParentOfType<AIE::DeviceOp>();
  if (!device) {
    // this check is redudnant with the HasParent trait, but can't hurt
    (*this)->emitOpError() << "must be inside AIE device operation.";
    return failure();
  }
  return success();
}

LogicalResult AIEX::RuntimeSequenceOp::verifyBeforeMaterialization() {
  // Check that all symbol references within the runtime sequence
  // are either to ShimDMAAllocationOp, DeviceOp or another RuntimeSequenceOp;
  // these are the only symbols that can be lowered with the NPU passes
  auto result = (*this)->walk([&](Operation *op) {
    for (NamedAttribute namedAttr : op->getAttrs()) {
      Attribute attr = namedAttr.getValue();
      auto walkResult = attr.walk([&](SymbolRefAttr symbolRef) {
        Operation *symbolDefOp = SymbolTable::lookupNearestSymbolFrom(*this, symbolRef);
        if (symbolDefOp && 
            !llvm::isa<AIE::ShimDMAAllocationOp>(symbolDefOp) && 
            !llvm::isa<AIE::DeviceOp>(symbolDefOp) &&
            !llvm::isa<AIEX::RuntimeSequenceOp>(symbolDefOp)) {
          op->emitOpError() << "references symbol '" 
                            << symbolRef.getRootReference().getValue()
                            << "' which must be either a ShimDMAAllocationOp, DeviceOp or RuntimeSequenceOp, but got: "
                            << symbolDefOp->getName().getStringRef();
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (walkResult.wasInterrupted()) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    return failure();
  }

  return success();
}

AIEX::RuntimeSequenceOp
AIEX::RuntimeSequenceOp::getForSymbolInDevice(AIE::DeviceOp deviceOp,
                                              llvm::StringRef symbol) {
  AIEX::RuntimeSequenceOp runtimeSequenceOp;
  if (!symbol.size()) {
    runtimeSequenceOp = *deviceOp.getOps<AIEX::RuntimeSequenceOp>().begin();
  } else {
    Operation *maybeRuntimeSequenceOp =
        mlir::SymbolTable::lookupSymbolIn(deviceOp, symbol);
    if (!maybeRuntimeSequenceOp) {
      return nullptr;
    }
    runtimeSequenceOp =
        llvm::dyn_cast<AIEX::RuntimeSequenceOp>(maybeRuntimeSequenceOp);
  }
  return runtimeSequenceOp;
}

AIEX::RuntimeSequenceOp
AIEX::RuntimeSequenceOp::getForSymbolInDeviceOrError(AIE::DeviceOp deviceOp,
                                                     llvm::StringRef symbol) {
  AIEX::RuntimeSequenceOp runtimeSequenceOp =
      getForSymbolInDevice(deviceOp, symbol);
  if (!runtimeSequenceOp) {
    if (!symbol.empty()) {
      deviceOp.emitError("No such runtime sequence: ") << symbol;
    } else {
      deviceOp.emitError("No runtime sequence in device");
    }
  }
  return runtimeSequenceOp;
}

//===----------------------------------------------------------------------===//