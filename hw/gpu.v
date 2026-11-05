module gpu #(parameter UNROLL = 1, parameter MAXQ = 31) (
  input clk,
  input        go,
  input [11:0] px, py,
  input [5:0]  ang,
  input [4:0]  dlo, dhi,
  input [4:0]  ddist,
  output       busy,
  output [32*5-1:0] dist_o,
  output reg [31:0] frame_demon = 0
);
  function wall(input [3:0] x, input [3:0] y);
    reg [15:0] row;
    begin
      case (y)
        4'd0, 4'd15: row = 16'hFFFF;
        4'd1:  row = 16'b1000000000000001;
        4'd2:  row = 16'b1011100001110001;
        4'd3:  row = 16'b1010000000010001;
        4'd4:  row = 16'b1010001100010001;
        4'd5:  row = 16'b1000001100000001;
        4'd6:  row = 16'b1000000000011101;
        4'd7:  row = 16'b1001110000000101;
        4'd8:  row = 16'b1000000000000101;
        4'd9:  row = 16'b1000000011000001;
        4'd10: row = 16'b1011000011000001;
        4'd11: row = 16'b1001000000001101;
        4'd12: row = 16'b1001000000001001;
        4'd13: row = 16'b1000000000000001;
        default: row = 16'hFFFF;
      endcase
      wall = row[x];
    end
  endfunction

  function signed [7:0] cosq(input [5:0] a);
    reg signed [7:0] t [0:15];
    begin
      t[0]=64; t[1]=64; t[2]=63; t[3]=61; t[4]=59; t[5]=57; t[6]=54; t[7]=50;
      t[8]=45; t[9]=40; t[10]=36; t[11]=30; t[12]=24; t[13]=19; t[14]=12; t[15]=6;
      case (a[5:4])
        2'd0: cosq = t[a[3:0]];
        2'd1: cosq = (a[3:0] == 0) ? 0 : -t[5'd16 - a[3:0]];
        2'd2: cosq = -t[a[3:0]];
        default: cosq = (a[3:0] == 0) ? 0 : t[5'd16 - a[3:0]];
      endcase
    end
  endfunction

  function signed [7:0] sinq(input [5:0] a);
    begin sinq = cosq(a - 6'd16); end
  endfunction

  reg running = 0;
  assign busy = running;
  reg [4:0] stepn = 0;
  wire [31:0] demon;
  genvar c;
  generate for (c = 0; c < 32; c = c + 1) begin : col
    wire [5:0] ra = ang + 6'd16 - c[5:0];
    wire signed [7:0] dx = cosq(ra), dy = sinq(ra);
    reg [11:0] rx = 0, ry = 0;
    reg [4:0] q = 0;
    reg hit = 0;
    reg [11:0] x [0:UNROLL]; reg [11:0] y [0:UNROLL]; reg h [0:UNROLL]; reg [4:0] qq [0:UNROLL];
    integer u;
    always @* begin
      x[0] = rx; y[0] = ry; h[0] = hit; qq[0] = q;
      for (u = 0; u < UNROLL; u = u + 1) begin
        if (!h[u]) begin
          x[u+1] = x[u] + {{4{dx[7]}}, dx};
          y[u+1] = y[u] + {{4{dy[7]}}, dy};
          qq[u+1] = qq[u] + 5'd1;
          h[u+1] = wall(x[u+1][11:8], y[u+1][11:8]) | (qq[u] == MAXQ - 1);
        end else begin
          x[u+1] = x[u]; y[u+1] = y[u]; qq[u+1] = qq[u]; h[u+1] = 1;
        end
      end
    end
    always @(posedge clk) begin
      if (go) begin rx <= px; ry <= py; q <= 0; hit <= 0; end
      else if (running) begin rx <= x[UNROLL]; ry <= y[UNROLL]; q <= qq[UNROLL]; hit <= h[UNROLL]; end
    end
    assign dist_o[c*5 +: 5] = q;
    assign demon[c] = (c >= dlo) && (c <= dhi) && (ddist != 0) && !running;
  end endgenerate

  always @(posedge clk) begin
    if (go) begin running <= 1; stepn <= 0; end
    else if (running) begin stepn <= stepn + 1; if (stepn == (MAXQ + UNROLL - 1) / UNROLL) running <= 0; end
    if (!running && !go) frame_demon <= demon;
  end
endmodule
