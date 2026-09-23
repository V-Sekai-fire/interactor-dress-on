/*
 * riscv64 (LP64D) context switch for guest/fiber/fiber.h.
 *
 * fiber_ctx layout (208 bytes): ra @0, sp @8, s0-s11 @16..104, fs0-fs11
 * @112..200. Only the callee-saved registers are saved: fiber_switch is an
 * ordinary call, so the caller has already spilled everything else. tp and
 * gp are shared by every fiber (one hart, one TLS block) and are not touched.
 */
	.text
	.align 2

	.globl fiber_switch
	.type fiber_switch, @function
/* void fiber_switch(fiber_ctx *from = a0, const fiber_ctx *to = a1) */
fiber_switch:
	.cfi_startproc
	sd ra, 0(a0)
	sd sp, 8(a0)
	sd s0, 16(a0)
	sd s1, 24(a0)
	sd s2, 32(a0)
	sd s3, 40(a0)
	sd s4, 48(a0)
	sd s5, 56(a0)
	sd s6, 64(a0)
	sd s7, 72(a0)
	sd s8, 80(a0)
	sd s9, 88(a0)
	sd s10, 96(a0)
	sd s11, 104(a0)
	fsd fs0, 112(a0)
	fsd fs1, 120(a0)
	fsd fs2, 128(a0)
	fsd fs3, 136(a0)
	fsd fs4, 144(a0)
	fsd fs5, 152(a0)
	fsd fs6, 160(a0)
	fsd fs7, 168(a0)
	fsd fs8, 176(a0)
	fsd fs9, 184(a0)
	fsd fs10, 192(a0)
	fsd fs11, 200(a0)

	ld ra, 0(a1)
	ld sp, 8(a1)
	ld s0, 16(a1)
	ld s1, 24(a1)
	ld s2, 32(a1)
	ld s3, 40(a1)
	ld s4, 48(a1)
	ld s5, 56(a1)
	ld s6, 64(a1)
	ld s7, 72(a1)
	ld s8, 80(a1)
	ld s9, 88(a1)
	ld s10, 96(a1)
	ld s11, 104(a1)
	fld fs0, 112(a1)
	fld fs1, 120(a1)
	fld fs2, 128(a1)
	fld fs3, 136(a1)
	fld fs4, 144(a1)
	fld fs5, 152(a1)
	fld fs6, 160(a1)
	fld fs7, 168(a1)
	fld fs8, 176(a1)
	fld fs9, 184(a1)
	fld fs10, 192(a1)
	fld fs11, 200(a1)
	ret
	.cfi_endproc
	.size fiber_switch, .-fiber_switch

	.globl fiber_trampoline
	.type fiber_trampoline, @function
/* First frame of a fiber: s0 = Fiber*, s1 = fiber_main_c. ra is marked
 * undefined so an unwinder that walks this far stops here instead of
 * wandering into whichever stack last resumed us. */
fiber_trampoline:
	.cfi_startproc
	.cfi_undefined ra
	mv a0, s0
	jalr s1
	unimp
	.cfi_endproc
	.size fiber_trampoline, .-fiber_trampoline

	.section .note.GNU-stack,"",@progbits
