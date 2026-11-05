module fly_fpu(input clk, output [31:0] io_out, output [31:0] pc, output halted, output io_wr);
  wire [31:0] io_in;
  neural_port port(.clk(clk), .v(io_in));
  fpu #(.AW(8)) core(.clk(clk), .io_in(io_in), .io_out_o(io_out), .pc_o(pc), .halted_o(halted), .io_wr_o(io_wr));
endmodule

module fly_fpu_gpu(input clk, output [31:0] io_out, output [31:0] pc, output halted, output io_wr,
                   output [31:0] frame_demon, output [159:0] dist, output busy, output [31:0] cam, output [31:0] dem);
  wire [31:0] io_in;
  wire go;
  neural_port port(.clk(clk), .v(io_in));
  fpu #(.AW(8)) core(.clk(clk), .io_in(io_in), .io_out_o(io_out), .pc_o(pc), .halted_o(halted), .io_wr_o(io_wr),
    .cam_o(cam), .dem_o(dem), .go_o(go), .aux_in({busy, 26'd0, dist[15*5 +: 5]}));
  gpu g(.clk(clk), .go(go), .px(cam[11:0]), .py(cam[23:12]), .ang(cam[29:24]),
        .dlo(dem[4:0]), .dhi(dem[9:5]), .ddist(dem[14:10]), .busy(busy), .dist_o(dist), .frame_demon(frame_demon));
endmodule

module fpu_ext(input clk, input [31:0] ext_rdata, output [15:0] ext_widx, output ext_we, output [31:0] ext_wdata,
               output [31:0] io_out, output [31:0] pc, output halted, output io_wr);
  fpu #(.AW(14)) core(.clk(clk), .io_in(32'd0), .io_out_o(io_out), .pc_o(pc), .halted_o(halted), .io_wr_o(io_wr),
    .ext_rdata(ext_rdata), .ext_widx(ext_widx), .ext_we(ext_we), .ext_wdata(ext_wdata));
endmodule
