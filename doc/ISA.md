## Registers

| Name  | Alias  |  Index  | Saver  |
| ----- | ------ | ------- | ------ |
| `pc`  | -      | -       | -      |
| -     |        |         |        |
| `r0 ` | `zero` | `00000` | -      |
| `r1 ` | `ra  ` | `00001` | caller |
| `r2 ` | `sp  ` | `00010` | callee |
| `r3 ` | `a0  ` | `00011` | caller |
| `r4 ` | `a1  ` | `00100` | caller |
| `r5 ` | `a2  ` | `00101` | caller |
| `r6 ` | `a3  ` | `00110` | caller |
| `r7 ` | `a4  ` | `00111` | caller |
| `r8 ` | `a5  ` | `01000` | caller |
| `r9 ` | `a6  ` | `01001` | caller |
| `r10` | `a7  ` | `01010` | caller |
| `r11` | `t0  ` | `01011` | caller |
| `r12` | `t1  ` | `01100` | caller |
| `r13` | `t2  ` | `01101` | caller |
| `r14` | `t3  ` | `01110` | caller |
| `r15` | `t4  ` | `01111` | caller |
| `r16` | `t5  ` | `10000` | caller |
| `r17` | `t6  ` | `10001` | caller |
| `r18` | `t7  ` | `10010` | caller |
| `r19` | `s0  ` | `10011` | callee |
| `r20` | `s1  ` | `10100` | callee |
| `r21` | `s2  ` | `10101` | callee |
| `r22` | `s3  ` | `10110` | callee |
| `r23` | `s4  ` | `10111` | callee |
| `r24` | `s5  ` | `11000` | callee |
| `r25` | `s6  ` | `11001` | callee |
| `r26` | `s7  ` | `11010` | callee |
| `r27` | `s8  ` | `11011` | callee |
| `r28` | `s9  ` | `11100` | callee |
| `r29` | `s10 ` | `11101` | callee |
| `r30` | `s11 ` | `11110` | callee |
| `r31` | `s12 ` | `11111` | callee |


## Flags
| Name | Description |
| ---- | ----------- |
| `C`  | Carry       |
| `Z`  | Zero        |
| `S`  | Sign        |
| `O`  | Overflow    |
| `K`  | Kernel mode |


## Instructions

| Mnemonic + operands | Bit pattern                             | Affected flags | Operation |
| ------------------- | --------------------------------------- | -------------- | --------- |
| `NOP              ` | `----------_-----_-----_-----_--00_000` | `-           ` | -         |
| `BRK              ` | `----------_-----_-----_-----_--01_000` | `-           ` | Pauses emulation (behaves like NOP in hardware) |
| `HLT              ` | `----------_-----_-----_-----_--10_000` | `-           ` | Stops emulation gracefully (behaves like NOP in hardware) |
| `ERR              ` | `----------_-----_-----_-----_--11_000` | `-           ` | Stops emulation with an error (behaves like NOP in hardware) |
| -                   |                                         |                |           |
| `ADD     d, l, r  ` | `----------_rrrrr_lllll_ddddd_0000_001` | `C, Z, S, O  ` | `d = l + r` |
| `ADDC    d, l, r  ` | `----------_rrrrr_lllll_ddddd_0001_001` | `C, Z, S, O  ` | `d = l + r + C` |
| `SUB     d, l, r  ` | `----------_rrrrr_lllll_ddddd_0010_001` | `C, Z, S, O  ` | `d = l - r` |
| `SUBB    d, l, r  ` | `----------_rrrrr_lllll_ddddd_0011_001` | `C, Z, S, O  ` | `d = l - r - 1 + C` |
| `AND     d, l, r  ` | `----------_rrrrr_lllll_ddddd_0100_001` | `Z           ` | `d = l & r` |
| `OR      d, l, r  ` | `----------_rrrrr_lllll_ddddd_0101_001` | `Z           ` | `d = l \| r` |
| `XOR     d, l, r  ` | `----------_rrrrr_lllll_ddddd_0110_001` | `Z           ` | `d = l ^ r` |
| `SHL     d, l, r  ` | `----------_rrrrr_lllll_ddddd_0111_001` | `Z           ` | `d = l << r` |
| `LSR     d, l, r  ` | `----------_rrrrr_lllll_ddddd_1000_001` | `Z           ` | `d = l >> r` |
| `ASR     d, l, r  ` | `----------_rrrrr_lllll_ddddd_1001_001` | `Z           ` | `d = l >>> r` |
| `MUL     d, l, r  ` | `----------_rrrrr_lllll_ddddd_1010_001` | `Z           ` | `d = (l * r)[31..0]` |
| `MULHUU  d, l, r  ` | `----------_rrrrr_lllll_ddddd_1011_001` | `Z           ` | `d = (l * r)[63..32]` |
| `MULHSS  d, l, r  ` | `----------_rrrrr_lllll_ddddd_1100_001` | `Z           ` | `d = (l * r)[63..32]` |
| `MULHSU  d, l, r  ` | `----------_rrrrr_lllll_ddddd_1101_001` | `Z           ` | `d = (l * r)[63..32]` |
| `CSUB    d, l, r  ` | `----------_rrrrr_lllll_ddddd_1110_001` | `C, Z        ` | `d = (l >= r) ? l - r : l` |
| `SLC     d, l     ` | `----------_-----_lllll_ddddd_1111_001` | `C, Z        ` | `d = (l << 1) + C` |
| -                   |                                         |                |           |
| `ADD     d, l, v  ` | `vvvvvvvvvv_vvvvv_lllll_ddddd_0000_010` | `C, Z, S, O  ` | `d = l + v` |
| `ADDC    d, l, v  ` | `vvvvvvvvvv_vvvvv_lllll_ddddd_0001_010` | `C, Z, S, O  ` | `d = l + v + C` |
| `SUB     d, l, v  ` | `vvvvvvvvvv_vvvvv_lllll_ddddd_0010_010` | `C, Z, S, O  ` | `d = l - v` |
| `SUBB    d, l, v  ` | `vvvvvvvvvv_vvvvv_lllll_ddddd_0011_010` | `C, Z, S, O  ` | `d = l - v - 1 + C` |
| `AND     d, l, v  ` | `vvvvvvvvvv_vvvvv_lllll_ddddd_0100_010` | `Z           ` | `d = l & v` |
| `OR      d, l, v  ` | `vvvvvvvvvv_vvvvv_lllll_ddddd_0101_010` | `Z           ` | `d = l \| v` |
| `XOR     d, l, v  ` | `vvvvvvvvvv_vvvvv_lllll_ddddd_0110_010` | `Z           ` | `d = l ^ v` |
| `SHL     d, l, v  ` | `vvvvvvvvvv_vvvvv_lllll_ddddd_0111_010` | `Z           ` | `d = l << v` |
| `LSR     d, l, v  ` | `vvvvvvvvvv_vvvvv_lllll_ddddd_1000_010` | `Z           ` | `d = l >> v` |
| `ASR     d, l, v  ` | `vvvvvvvvvv_vvvvv_lllll_ddddd_1001_010` | `Z           ` | `d = l >>> v` |
| `MUL     d, l, v  ` | `vvvvvvvvvv_vvvvv_lllll_ddddd_1010_010` | `Z           ` | `d = (l * v)[31..0]` |
| `MULHUU  d, l, v  ` | `vvvvvvvvvv_vvvvv_lllll_ddddd_1011_010` | `Z           ` | `d = (l * v)[63..32]` |
| `MULHSS  d, l, v  ` | `vvvvvvvvvv_vvvvv_lllll_ddddd_1100_010` | `Z           ` | `d = (l * v)[63..32]` |
| `MULHSU  d, l, v  ` | `vvvvvvvvvv_vvvvv_lllll_ddddd_1101_010` | `Z           ` | `d = (l * v)[63..32]` |
| `CSUB    d, l, v  ` | `vvvvvvvvvv_vvvvv_lllll_ddddd_1110_010` | `C, Z        ` | `d = (l >= v) ? l - v : l` |
| `SLC     d, l     ` | `----------_-----_lllll_ddddd_1111_010` | `C, Z        ` | `d = (l << 1) + C` |
| -                   |                                         |                |           |
| `LD      d, [s, v]` | `vvvvvvvvvv_vvvvv_sssss_ddddd_0-00_011` | `-           ` | `d = mem[s + v]` |
| `LD8     d, [s, v]` | `vvvvvvvvvv_vvvvv_sssss_ddddd_0001_011` | `-           ` | `d = (u8)mem[s + v]` |
| `LD8S    d, [s, v]` | `vvvvvvvvvv_vvvvv_sssss_ddddd_0101_011` | `-           ` | `d = (s8)mem[s + v]` |
| `LD16    d, [s, v]` | `vvvvvvvvvv_vvvvv_sssss_ddddd_0010_011` | `-           ` | `d = (u16)mem[s + v]` |
| `LD16S   d, [s, v]` | `vvvvvvvvvv_vvvvv_sssss_ddddd_0110_011` | `-           ` | `d = (s16)mem[s + v]` |
| `IN      d, [s, v]` | `vvvvvvvvvv_vvvvv_sssss_ddddd_0-11_011` | `-           ` | `d = io[s + v]` |
| -                   |                                         |                |           |
| `ST      [d, v], s` | `vvvvvvvvvv_vvvvv_sssss_ddddd_1-00_011` | `-           ` | `mem[d + v] = s` |
| `ST8     [d, v], s` | `vvvvvvvvvv_vvvvv_sssss_ddddd_1-01_011` | `-           ` | `mem[d + v] = (i8)s` |
| `ST16    [d, v], s` | `vvvvvvvvvv_vvvvv_sssss_ddddd_1-10_011` | `-           ` | `mem[d + v] = (i16)s` |
| `OUT     [d, v], s` | `vvvvvvvvvv_vvvvv_sssss_ddddd_1-11_011` | `-           ` | `io[d + v] = s` |
| -                   |                                         |                |           |
| `JMP     s, v     ` | `vvvvvvvvvv_vvvvv_sssss_-----_---0_100` | `-           ` | `pc = s + v` |
| `JMP     [s, v]   ` | `vvvvvvvvvv_vvvvv_sssss_-----_---1_100` | `-           ` | `pc = mem[s + v]` |
| -                   |                                         |                |           |
| `BR.C    v        ` | `vvvvvvvvvv_vvvvv_vvvvv_-----_0001_101` | `-           ` | `if C then pc += v` |
| `BR.Z    v        ` | `vvvvvvvvvv_vvvvv_vvvvv_-----_0010_101` | `-           ` | `if Z then pc += v` |
| `BR.S    v        ` | `vvvvvvvvvv_vvvvv_vvvvv_-----_0011_101` | `-           ` | `if S then pc += v` |
| `BR.O    v        ` | `vvvvvvvvvv_vvvvv_vvvvv_-----_0100_101` | `-           ` | `if O then pc += v` |
| `BR.NC   v        ` | `vvvvvvvvvv_vvvvv_vvvvv_-----_0101_101` | `-           ` | `if !C then pc += v` |
| `BR.NZ   v        ` | `vvvvvvvvvv_vvvvv_vvvvv_-----_0110_101` | `-           ` | `if !Z then pc += v` |
| `BR.NS   v        ` | `vvvvvvvvvv_vvvvv_vvvvv_-----_0111_101` | `-           ` | `if !S then pc += v` |
| `BR.NO   v        ` | `vvvvvvvvvv_vvvvv_vvvvv_-----_1000_101` | `-           ` | `if !O then pc += v` |
| `BR.U.LE v        ` | `vvvvvvvvvv_vvvvv_vvvvv_-----_1001_101` | `-           ` | `if !C \|\| Z then pc += v` |
| `BR.U.G  v        ` | `vvvvvvvvvv_vvvvv_vvvvv_-----_1010_101` | `-           ` | `if C && !Z then pc += v` |
| `BR.S.L  v        ` | `vvvvvvvvvv_vvvvv_vvvvv_-----_1011_101` | `-           ` | `if S != O then pc += v` |
| `BR.S.GE v        ` | `vvvvvvvvvv_vvvvv_vvvvv_-----_1100_101` | `-           ` | `if S == O then pc += v` |
| `BR.S.LE v        ` | `vvvvvvvvvv_vvvvv_vvvvv_-----_1101_101` | `-           ` | `if Z \|\| (S != O) then pc += v` |
| `BR.S.G  v        ` | `vvvvvvvvvv_vvvvv_vvvvv_-----_1110_101` | `-           ` | `if !Z && (S == O) then pc += v` |
| `BRA     v        ` | `vvvvvvvvvv_vvvvv_vvvvv_-----_1111_101` | `-           ` | `pc += v` |
| -                   |                                         |                |           |
| `LDUI    d, v     ` | `vvvvvvvvvv_vvvvv_vvvvv_ddddd_---0_110` | `-           ` | `d = v` |
| `ADDPCUI d, v     ` | `vvvvvvvvvv_vvvvv_vvvvv_ddddd_---1_110` | `-           ` | `d = pc + v` |
| -                   |                                         |                |           |
| `SYS              ` | `----------_-----_-----_-----_---0_111` | `K           ` | `K = 1, pc = 0x00007FF0` |
| `CLRK             ` | `----------_-----_-----_-----_---1_111` | `K           ` | `K = 0` |


## Immediate Encodings

![immediates](Immediates.png)


## Pseudo-Instructions

| Mnemonic + operands | Actual instruction emitted | Affected flags | Operation |
| ------------------- | -------------------------- | -------------- | --------- |
| `MOV     d, s     ` | `OR    d, zero, s        ` | `Z           ` | `d = s` |
| `LDI     d, v     ` | `OR    d, zero, v        ` | `Z           ` | `d = v` |
| -                   |                            |                |           |
| `CMP     l, r     ` | `SUB   zero, l, r        ` | `C, Z, S, O  ` | `-` |
| `CMP     l, v     ` | `SUB   zero, l, v        ` | `C, Z, S, O  ` | `-` |
| -                   |                            |                |           |
| `BIT     l, r     ` | `AND   zero, l, r        ` | `Z           ` | `-` |
| `BIT     l, v     ` | `AND   zero, l, v        ` | `Z           ` | `-` |
| -                   |                            |                |           |
| `TEST    s        ` | `OR    zero, s, zero     ` | `Z           ` | `-` |
| -                   |                            |                |           |
| `INC     d        ` | `ADD   d, d, 1           ` | `C, Z, S, O  ` | `d = d + 1` |
| `INCC    d        ` | `ADDC  d, d, 0           ` | `C, Z, S, O  ` | `d = d + C` |
| `DEC     d        ` | `SUB   d, d, 1           ` | `C, Z, S, O  ` | `d = d - 1` |
| `DECB    d        ` | `SUBB  d, d, 0           ` | `C, Z, S, O  ` | `d = d - 1 + C` |
| -                   |                            |                |           |
| `NEG     d, s     ` | `SUB   d, zero, s        ` | `C, Z, S, O  ` | `d = -d` |
| `NEGB    d, s     ` | `SUBB  d, zero, s        ` | `C, Z, S, O  ` | `d = -d - 1 + C` |
| -                   |                            |                |           |
| `NOT     d, s     ` | `XOR   d, s, -1          ` | `Z           ` | `d = !d` |
| -                   |                            |                |           |
| `LD      d, [s]   ` | `LD    d, [s, 0]         ` | `-           ` | `d = mem[s]` |
| `LD      d, [v]   ` | `LD    d, [zero, v]      ` | `-           ` | `d = mem[v]` |
| `LD8     d, [s]   ` | `LD8   d, [s, 0]         ` | `-           ` | `d = (u8)mem[s]` |
| `LD8     d, [v]   ` | `LD8   d, [zero, v]      ` | `-           ` | `d = (u8)mem[v]` |
| `LD8S    d, [s]   ` | `LD8S  d, [s, 0]         ` | `-           ` | `d = (s8)mem[s]` |
| `LD8S    d, [v]   ` | `LD8S  d, [zero, v]      ` | `-           ` | `d = (s8)mem[v]` |
| `LD16    d, [s]   ` | `LD16  d, [s, 0]         ` | `-           ` | `d = (u16)mem[s]` |
| `LD16    d, [v]   ` | `LD16  d, [zero, v]      ` | `-           ` | `d = (u16)mem[v]` |
| `LD16S   d, [s]   ` | `LD16S d, [s, 0]         ` | `-           ` | `d = (s16)mem[s]` |
| `LD16S   d, [v]   ` | `LD16S d, [zero, v]      ` | `-           ` | `d = (s16)mem[v]` |
| `IN      d, [s]   ` | `IN    d, [s, 0]         ` | `-           ` | `d = io[s]` |
| `IN      d, [v]   ` | `IN    d, [zero, v]      ` | `-           ` | `d = io[v]` |
| -                   |                            |                |           |
| `ST      [d], s   ` | `ST    [d, 0], s         ` | `-           ` | `mem[d] = s` |
| `ST      [v], s   ` | `ST    [zero, v], s      ` | `-           ` | `mem[v] = s` |
| `ST8     [d], s   ` | `ST8   [d, 0], s         ` | `-           ` | `mem[d] = (i8)s` |
| `ST8     [v], s   ` | `ST8   [zero, v], s      ` | `-           ` | `mem[v] = (i8)s` |
| `ST16    [d], s   ` | `ST16  [d, 0], s         ` | `-           ` | `mem[d] = (i16)s` |
| `ST16    [v], s   ` | `ST16  [zero, v], s      ` | `-           ` | `mem[v] = (i16)s` |
| `OUT     [d], s   ` | `OUT   [d, 0], s         ` | `-           ` | `io[d] = s` |
| `OUT     [v], s   ` | `OUT   [zero, v], s      ` | `-           ` | `io[v] = s` |
| -                   |                            |                |           |
| `JMP     s        ` | `JMP   s, 0              ` | `-           ` | `pc = s` |
| `JMP     v        ` | `JMP   zero, v           ` | `-           ` | `pc = v` |
| `JMP     [s]      ` | `JMP   [s, 0]            ` | `-           ` | `pc = mem[s]` |
| `JMP     [v]      ` | `JMP   [zero, v]         ` | `-           ` | `pc = mem[v]` |
| -                   |                            |                |           |
| `BR.EQ   v        ` | `BR.Z  v                 ` | `-           ` | `if Z then pc += v` |
| `BR.NEQ  v        ` | `BR.NZ v                 ` | `-           ` | `if !Z then pc += v` |
| `BR.U.L  v        ` | `BR.NC v                 ` | `-           ` | `if !C then pc += v` |
| `BR.U.GE v        ` | `BR.C  v                 ` | `-           ` | `if C then pc += v` |
