(* blackbox *)
module neural_ram #(parameter AW = 8) (input clk, input [AW-1:0] addr, output [31:0] rdata, input we, input [31:0] wdata);
endmodule

(* blackbox *)
module neural_port(input clk, output [31:0] v);
endmodule
