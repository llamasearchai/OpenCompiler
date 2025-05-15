"""
Core compiler infrastructure for CPU-targeted ML optimizations.
"""

import mlir
from mlir.ir import *
from mlir.dialects import func, arith, memref, linalg
import logging
import platform
from enum import Enum, auto

logger = logging.getLogger(__name__)

class TargetArch(Enum):
    X86_64 = auto()
    X86_64_AVX2 = auto()
    X86_64_AVX512 = auto()
    ARM64 = auto()
    ARM64_NEON = auto()
    ARM64_SVE = auto()
    APPLE_SILICON = auto()  # Special case for Apple M-series

    @staticmethod
    def detect_host():
        """Detect the host architecture and capabilities."""
        # This detection logic prioritizes specific Apple Silicon identification
        # and then falls back to more generic CPU feature detection using cpuinfo.
        # For x86, it checks for AVX capabilities to enable relevant vectorization.
        # For ARM, it distinguishes between generic ARM64, NEON-capable, SVE-capable,
        # and specific Apple Silicon.
        machine = platform.machine().lower()
        system = platform.system().lower()
        
        if machine in ['x86_64', 'amd64']:
            # Check for AVX512/AVX2 support
            try:
                import cpuinfo
                info = cpuinfo.get_cpu_info()
                if 'avx512f' in info.get('flags', []):
                    return TargetArch.X86_64_AVX512
                elif 'avx2' in info.get('flags', []):
                    return TargetArch.X86_64_AVX2
                else:
                    return TargetArch.X86_64
            except ImportError:
                return TargetArch.X86_64
        
        elif machine in ['arm64', 'aarch64']:
            if system == 'darwin':
                # Apple Silicon detection
                model = platform.processor()
                if 'apple m' in model.lower():
                    return TargetArch.APPLE_SILICON
            
            # Generic ARM64 detection
            try:
                import cpuinfo
                info = cpuinfo.get_cpu_info()
                if 'asimd' in info.get('flags', []):  # NEON
                    return TargetArch.ARM64_NEON
                elif 'sve' in info.get('flags', []):  # SVE
                    return TargetArch.ARM64_SVE
                else:
                    return TargetArch.ARM64
            except ImportError:
                return TargetArch.ARM64
        
        # Default fallback
        return TargetArch.X86_64


class CPUCompiler:
    """
    Core compiler class for CPU-targeted ML optimizations.
    
    This class handles the transformation of high-level ML operations
    into optimized CPU code through MLIR-based compilation pipeline.
    Support for x86-64 and ARM64 architectures, including Apple Silicon.
    """
    
    def __init__(self, target_arch=None, optimization_level=3):
        """
        Initialize the CPU compiler.
        
        Args:
            target_arch: Target CPU architecture (TargetArch enum or string)
            optimization_level: Optimization level (0-3)
        """
        # Handle target architecture
        if target_arch is None:
            self.target_arch = TargetArch.detect_host()
        elif isinstance(target_arch, str):
            # Convert string to enum
            try:
                self.target_arch = TargetArch[target_arch.upper()]
            except KeyError:
                logger.warning(f"Unknown architecture: {target_arch}, defaulting to host detection")
                self.target_arch = TargetArch.detect_host()
        else:
            self.target_arch = target_arch
            
        self.optimization_level = optimization_level
        self.context = Context()
        self.register_dialects()
        logger.info(f"Initialized CPU compiler for {self.target_arch.name} with O{optimization_level}")
        
    def register_dialects(self):
        """Register all required MLIR dialects."""
        # MLIR requires dialects to be registered in the context before they can be used
        # in parsing or constructing IR.
        with self.context:
            func.register_dialect(self.context)
            arith.register_dialect(self.context)
            memref.register_dialect(self.context)
            linalg.register_dialect(self.context)
            # Register architecture-specific dialects
            self._register_arch_specific_dialects()
        
    def _register_arch_specific_dialects(self):
        """Register architecture-specific dialects based on target."""
        # This method demonstrates conditional dialect registration based on the
        # target architecture. For a production compiler, these `extras` dialects
        # would need to be robustly available in the MLIR build.
        # Common dialects for all architectures
        with self.context:
            from mlir.dialects import vector, scf, gpu
            vector.register_dialect(self.context)
            scf.register_dialect(self.context)
            
            # Architecture-specific dialects
            if self.target_arch in [TargetArch.X86_64, TargetArch.X86_64_AVX2, TargetArch.X86_64_AVX512]:
                # X86-specific dialects and extensions
                try:
                    from mlir.extras.dialects import x86vector
                    x86vector.register_dialect(self.context)
                except ImportError:
                    logger.warning("x86vector dialect not available, falling back to generic vector ops")
            
            elif self.target_arch in [TargetArch.ARM64, TargetArch.ARM64_NEON, TargetArch.ARM64_SVE, TargetArch.APPLE_SILICON]:
                # ARM-specific dialects and extensions
                try:
                    from mlir.extras.dialects import arm_neon, arm_sve
                    arm_neon.register_dialect(self.context)
                    if self.target_arch in [TargetArch.ARM64_SVE]:
                        arm_sve.register_dialect(self.context)
                except ImportError:
                    logger.warning("ARM dialects not available, falling back to generic vector ops")
                
                if self.target_arch == TargetArch.APPLE_SILICON:
                    # Apple-specific optimizations
                    gpu.register_dialect(self.context)
                    try:
                        from mlir.extras.dialects import ane  # Apple Neural Engine
                        ane.register_dialect(self.context)
                        # NOTE: The 'ane' dialect is hypothetical here and would represent
                        # a custom dialect for targeting Apple Neural Engine operations directly from MLIR.
                    except ImportError:
                        logger.warning("Apple Neural Engine dialect (mlir.extras.dialects.ane) not available.")
    
    def compile(self, module_ir, output_format="llvm"):
        """
        Compile MLIR module to target format.
        
        Args:
            module_ir: MLIR module representation
            output_format: Target output format (llvm, object, assembly)
            
        Returns:
            Compiled output in requested format
        """
        with self.context:
            # Parse the input IR
            module = Module.parse(module_ir, self.context)
            
            # Apply the optimization pipeline
            from mlir.passmanager import PassManager
            pm = PassManager.parse(self._get_pipeline_string(), self.context)
            pm.run(module)
            
            # Lower to LLVM and compile to target format
            if output_format == "llvm":
                return self._lower_to_llvm(module)
            elif output_format == "object":
                return self._compile_to_object(module)
            elif output_format == "assembly":
                return self._compile_to_assembly(module)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
    
    def _get_pipeline_string(self):
        """Generate optimization pipeline based on target and opt level."""
        # This defines a basic, representative MLIR pass pipeline.
        # A production compiler would have a much more sophisticated pipeline, potentially
        # with multiple stages and dynamically chosen passes.
        # The order of passes matters significantly.
        pipeline = [
            # Standard canonicalization and cleanup passes.
            "canonicalize",
            "cse",
            "convert-linalg-to-loops",
            "lower-affine",
            "convert-scf-to-cf"
        ]
        
        # Architecture-specific optimizations
        if self.target_arch in [TargetArch.X86_64_AVX2, TargetArch.X86_64_AVX512]:
            # AVX-specific passes
            if self.optimization_level >= 2:
                pipeline.insert(0, "linalg-fusion")
                pipeline.append("x86vector-vectorize")
                if self.target_arch == TargetArch.X86_64_AVX512:
                    pipeline.append("x86vector-avx512-optimize")
                else:
                    pipeline.append("x86vector-avx2-optimize")
        
        elif self.target_arch in [TargetArch.ARM64_NEON, TargetArch.ARM64_SVE, TargetArch.APPLE_SILICON]:
            # ARM-specific passes
            if self.optimization_level >= 2:
                pipeline.insert(0, "linalg-fusion")
                pipeline.append("arm-vectorize")
                if self.target_arch == TargetArch.ARM64_NEON:
                    pipeline.append("arm-neon-optimize")
                elif self.target_arch == TargetArch.ARM64_SVE:
                    pipeline.append("arm-sve-optimize")
                elif self.target_arch == TargetArch.APPLE_SILICON:
                    # apple-ane-optimize is a conceptual pass placeholder.
                    # Real ANE targeting would involve complex pattern matching, subgraph outlining,
                    # and lowering to ANE-specific operations or runtime calls.
                    pipeline.append("apple-ane-optimize")
        
        # Generic optimizations based on level
        if self.optimization_level >= 3:
            pipeline.append("parallel-loops")
            pipeline.append("mem-optimize")
        
        return ",".join(pipeline)
    
    def _lower_to_llvm(self, module):
        """Lower MLIR module to LLVM IR."""
        # `lower_to_llvm_ir` is a utility function from MLIR's Python bindings
        # that runs the standard MLIR to LLVM IR conversion pipeline.
        from mlir.execution_engine import lower_to_llvm_ir
        return lower_to_llvm_ir(module, opt_level=self.optimization_level)
    
    def _compile_to_object(self, module):
        """Compile MLIR module to object code."""
        # Get LLVM IR
        llvm_ir = self._lower_to_llvm(module)
        
        # Use LLVM to compile to object code
        from llvmlite import binding as llvm
        
        # Initialize LLVM
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        
        # Create a module from the IR
        mod = llvm.parse_assembly(llvm_ir)
        mod.verify()
        
        # Create target machine
        target = llvm.Target.from_default_triple()
        
        # Configure CPU features based on architecture
        # These CPU names and features strings are crucial for LLVM to generate
        # optimized code for the specific microarchitecture.
        # They should align with LLVM's known targets.
        if self.target_arch == TargetArch.X86_64_AVX2:
            cpu = "x86-64-v3"  # AVX2 baseline
            features = "+avx2,+fma"
        elif self.target_arch == TargetArch.X86_64_AVX512:
            cpu = "x86-64-v4"  # AVX512 baseline
            features = "+avx512f,+avx512bw,+avx512cd,+avx512dq,+avx512vl"
        elif self.target_arch == TargetArch.APPLE_SILICON:
            cpu = "apple-m1" # Or a more specific M-series target if llvmlite/LLVM supports it
            features = "+v8.5a,+crypto,+dotprod,+fp16fml,+fullfp16,+neon" # Common M1 features
        elif self.target_arch in [TargetArch.ARM64, TargetArch.ARM64_NEON]:
            cpu = "generic"
            features = "+neon"
        elif self.target_arch == TargetArch.ARM64_SVE:
            cpu = "generic"
            features = "+sve"
        else:
            cpu = "generic"
            features = ""
        
        target_machine = target.create_target_machine(
            cpu=cpu,
            features=features,
            opt=self.optimization_level
        )
        
        # Compile to object code
        obj_code = target_machine.emit_object(mod)
        return obj_code

    def _compile_to_assembly(self, module):
        """Compile MLIR module to assembly."""
        # Get LLVM IR
        llvm_ir = self._lower_to_llvm(module)
        
        # Use LLVM to compile to assembly
        from llvmlite import binding as llvm
        
        # Initialize LLVM
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        
        # Create a module from the IR
        mod = llvm.parse_assembly(llvm_ir)
        mod.verify()
        
        # Create target machine with architecture-specific settings
        target = llvm.Target.from_default_triple()
        
        # Configure CPU features based on architecture (same as in _compile_to_object)
        # Duplication of this logic with _compile_to_object suggests refactoring
        # into a helper method if more compilation targets are added.
        if self.target_arch == TargetArch.X86_64_AVX2:
            cpu = "x86-64-v3"  # AVX2 baseline
            features = "+avx2,+fma"
        elif self.target_arch == TargetArch.X86_64_AVX512:
            cpu = "x86-64-v4"  # AVX512 baseline
            features = "+avx512f,+avx512bw,+avx512cd,+avx512dq,+avx512vl"
        elif self.target_arch == TargetArch.APPLE_SILICON:
            cpu = "apple-m1" # As above
            features = "+v8.5a,+crypto,+dotprod,+fp16fml,+fullfp16,+neon"
        elif self.target_arch in [TargetArch.ARM64, TargetArch.ARM64_NEON]:
            cpu = "generic"
            features = "+neon"
        elif self.target_arch == TargetArch.ARM64_SVE:
            cpu = "generic"
            features = "+sve"
        else:
            cpu = "generic"
            features = ""
        
        target_machine = target.create_target_machine(
            cpu=cpu,
            features=features,
            opt=self.optimization_level
        )
        
        # Generate assembly
        asm = target_machine.emit_assembly(mod)
        return asm 