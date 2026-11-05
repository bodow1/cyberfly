  .section .text.start
  .globl _start
_start:
  li sp, STACK_TOP
  call main
  ecall
1:
  j 1b
