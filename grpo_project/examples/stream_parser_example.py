from utils.stream_parser import StreamParser
import time

def simulate_stream_output():
    """模拟流式输出"""
    # 示例输出块
    chunks = [
        "让我思考一下这个问题...\n",
        "<think>\n",
        "首先，我们需要考虑输入信号的处理方式。",
        " 对于这个设计，我们应该使用同步复位。",
        "</think>\n\n",
        "现在开始编写代码：\n",
        "```verilog\n",
        "module example(",
        "  input clk,",
        "  input rst_n,",
        "  input [7:0] data_in,",
        "  output reg [7:0] data_out",
        ");",
        "  always @(posedge clk or negedge rst_n) begin",
        "    if (!rst_n) begin",
        "      data_out <= 8'h0;",
        "    end else begin",
        "      data_out <= data_in;",
        "    end",
        "  end",
        "endmodule\n",
        "```"
    ]
    
    parser = StreamParser()
    
    print("开始模拟流式输出...\n")
    for chunk in chunks:
        # 处理输出块
        processed_chunk, guidance = parser.process_chunk(chunk)
        
        # 打印处理后的块
        print(processed_chunk, end="", flush=True)
        
        # 如果需要引导，打印引导提示
        if guidance:
            print(f"\n[引导提示] {guidance}\n")
            
        # 模拟输出延迟
        time.sleep(0.5)
    
    # 打印最终状态信息
    print("\n输出完成！状态信息：")
    debug_info = parser.get_debug_info()
    for key, value in debug_info.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    simulate_stream_output() 