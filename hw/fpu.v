module fpu #(parameter AW = 8) (
  input clk,
  input  [31:0] io_in,
  output [31:0] io_out_o,
  output [31:0] pc_o,
  output        halted_o,
  output        io_wr_o
`ifdef GPU_PORT
  , output reg [31:0] cam_o = 0, output reg [31:0] dem_o = 0, output reg go_o = 0, input [31:0] aux_in
`endif
`ifdef EXT_MEM
  , input [31:0] ext_rdata, output [15:0] ext_widx, output ext_we, output [31:0] ext_wdata
`endif
);
  localparam MEMW = 1 << AW;
  reg [31:0] mem  [0:MEMW-1];
  reg [31:0] regs [0:31];
  reg [31:0] pc = 0, ir = 32'h00000013;
  reg exec = 0, halted = 0, io_wr = 0;
  reg [31:0] io_out = 0;
  integer i;
  initial begin
    for (i = 0; i < 32; i = i + 1) regs[i] = 0;
`ifndef NEURAL_RAM
    for (i = 0; i < MEMW; i = i + 1) mem[i] = 0;
`endif
  end
  assign io_out_o = io_out; assign pc_o = pc; assign halted_o = halted; assign io_wr_o = io_wr;

  wire [6:0] op = ir[6:0];
  wire [4:0] rd = ir[11:7], rs1 = ir[19:15], rs2 = ir[24:20];
  wire [2:0] f3 = ir[14:12];
  wire       f7 = ir[30];
  wire [31:0] imm_i = {{20{ir[31]}}, ir[31:20]};
  wire [31:0] imm_s = {{20{ir[31]}}, ir[31:25], ir[11:7]};
  wire [31:0] imm_b = {{19{ir[31]}}, ir[31], ir[7], ir[30:25], ir[11:8], 1'b0};
  wire [31:0] imm_u = {ir[31:12], 12'b0};
  wire [31:0] imm_j = {{11{ir[31]}}, ir[31], ir[19:12], ir[20], ir[30:21], 1'b0};
  wire is_lui = op == 7'b0110111, is_auipc = op == 7'b0010111, is_jal = op == 7'b1101111;
  wire is_jalr = op == 7'b1100111, is_br = op == 7'b1100011, is_ld = op == 7'b0000011;
  wire is_st = op == 7'b0100011, is_opi = op == 7'b0010011, is_op = op == 7'b0110011;
  wire is_sys = op == 7'b1110011;

  wire [31:0] r1 = (rs1 == 0) ? 32'd0 : regs[rs1];
  wire [31:0] r2 = (rs2 == 0) ? 32'd0 : regs[rs2];

  wire [31:0] b_op = is_op ? r2 : imm_i;
  wire        sub  = is_op & f7;
  wire [31:0] sum  = r1 + (sub ? ~b_op : b_op) + {31'd0, sub};
  wire [4:0]  sh   = b_op[4:0];
  wire        lt   = $signed(r1) < $signed(b_op);
  wire        ltu  = r1 < b_op;
  wire [31:0] sra  = $signed(r1) >>> sh;
  reg  [31:0] alu;
  always @* begin
    case (f3)
      3'd0: alu = sum;
      3'd1: alu = r1 << sh;
      3'd2: alu = {31'd0, lt};
      3'd3: alu = {31'd0, ltu};
      3'd4: alu = r1 ^ b_op;
      3'd5: alu = f7 ? sra : (r1 >> sh);
      3'd6: alu = r1 | b_op;
      default: alu = r1 & b_op;
    endcase
  end

  wire beq = r1 == r2;
  wire blt = $signed(r1) < $signed(r2);
  wire bltu = r1 < r2;
  reg take;
  always @* begin
    case (f3)
      3'd0: take = beq;  3'd1: take = !beq;
      3'd4: take = blt;  3'd5: take = !blt;
      3'd6: take = bltu; 3'd7: take = !bltu;
      default: take = 0;
    endcase
  end

  wire [31:0] maddr = r1 + (is_st ? imm_s : imm_i);
  wire        io_sel = maddr[31];
  wire [AW-1:0] widx = exec ? maddr[AW+1:2] : pc[AW+1:2];
  wire        mwe = exec & !halted & is_st & !io_sel;
`ifdef NEURAL_RAM
  wire [31:0] mrd;
  neural_ram #(.AW(AW)) ram(.clk(clk), .addr(widx), .rdata(mrd), .we(mwe), .wdata(merged));
`elsif EXT_MEM
  wire [31:0] mrd = ext_rdata;
  assign ext_widx = widx; assign ext_we = mwe; assign ext_wdata = merged;
`else
  wire [31:0] mrd = mem[widx];
`endif
  wire [1:0]  bo = maddr[1:0];
  reg [31:0] io_in_q = 0;
  always @(posedge clk) io_in_q <= io_in;
`ifdef GPU_PORT
  wire [31:0] word_in = io_sel ? (maddr[2] ? aux_in : io_in_q) : mrd;
`else
  wire [31:0] word_in = io_sel ? io_in_q : mrd;
`endif
  wire [31:0] shr = word_in >> {bo, 3'b000};
  reg  [31:0] ldv;
  always @* begin
    case (f3)
      3'd0: ldv = {{24{shr[7]}}, shr[7:0]};
      3'd1: ldv = {{16{shr[15]}}, shr[15:0]};
      3'd4: ldv = {24'd0, shr[7:0]};
      3'd5: ldv = {16'd0, shr[15:0]};
      default: ldv = word_in;
    endcase
  end
  wire [31:0] wmask = (f3 == 3'd0) ? (32'hFF << {bo, 3'b000}) :
                      (f3 == 3'd1) ? (32'hFFFF << {bo, 3'b000}) : 32'hFFFFFFFF;
  wire [31:0] wdat  = r2 << {bo, 3'b000};
  wire [31:0] merged = (mrd & ~wmask) | (wdat & wmask);

  wire [31:0] pc4 = pc + 32'd4;
  reg  [31:0] wbv;
  always @* begin
    if (is_lui) wbv = imm_u;
    else if (is_auipc) wbv = pc + imm_u;
    else if (is_jal | is_jalr) wbv = pc4;
    else if (is_ld) wbv = ldv;
    else wbv = alu;
  end
  wire wb_en = (is_lui | is_auipc | is_jal | is_jalr | is_ld | is_opi | is_op) & (rd != 0);
  wire [31:0] npc = is_jal ? pc + imm_j :
                    is_jalr ? ((r1 + imm_i) & ~32'd1) :
                    (is_br & take) ? pc + imm_b : pc4;

  always @(posedge clk) begin
    if (!halted) begin
      if (!exec) begin
        ir <= mrd;
        exec <= 1;
        io_wr <= 0;
`ifdef GPU_PORT
        go_o <= 0;
`endif
      end else begin
        exec <= 0;
        pc <= npc;
        if (wb_en) regs[rd] <= wbv;
`ifndef EXT_MEM
`ifndef NEURAL_RAM
        if (mwe) mem[widx] <= merged;
`endif
`endif
`ifdef GPU_PORT
        if (is_st & io_sel & maddr[3:2] == 2'd0) begin io_out <= r2; io_wr <= 1; end
        if (is_st & io_sel & maddr[3:2] == 2'd1) cam_o <= r2;
        if (is_st & io_sel & maddr[3:2] == 2'd2) begin dem_o <= r2; go_o <= r2[31]; end
`else
        if (is_st & io_sel) begin io_out <= r2; io_wr <= 1; end
`endif
        if (is_sys) halted <= 1;
      end
    end
  end
endmodule
