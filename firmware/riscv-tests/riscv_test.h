#define RVTEST_RV32U
#define RVTEST_RV64U
#define TESTNUM gp
#define RVTEST_CODE_BEGIN .text; .globl _start; _start: li TESTNUM, 0;
#define RVTEST_CODE_END
#define RVTEST_PASS li t6, 0x80000000; li a0, 1; sw a0, 0(t6); ecall;
#define RVTEST_FAIL li t6, 0x80000000; li a0, 0xBAD00000; or a0, a0, TESTNUM; sw a0, 0(t6); ecall;
#define RVTEST_DATA_BEGIN .data; .align 4;
#define RVTEST_DATA_END
