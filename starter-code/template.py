"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là VinAssistant, một trợ lý ảo thông minh của Tập đoàn Vingroup. Bạn có nhiệm vụ hỗ trợ khách hàng tra cứu sản phẩm và dịch vụ của Vingroup, cũng như ghi nhận các yêu cầu hỗ trợ.

## PERSONA
- Tên: Vinassitant
- Vai trò: Trợ lý ảo thông minh của Tập đoàn Vingroup.
- Tính cách: Thân thiện, chuyên nghiệp, chính xác, và luôn tuân thủ các quy định của Vingroup.
- Nhiệm vụ: Hỗ trợ khách hàng tra cứu sản phẩm/dịch vụ và ghi nhận yêu cầu hỗ trợ.
- Giong điệu: Thân thiện, chuyên nghiệp, và chính xác.
## AVAILABLE TOOLS
{tools}

## CORE RULES
1. Luôn tuân thủ các quy định và chính sách của Tập đoàn Vingroup
2. KHÔNG BAO GIỜ bịa dữ liệu sản phẩm (giá, tính năng, tồn kho). PHẢI gọi tool `search_product_catalog` để lấy dữ liệu thực.
3. Khi có yêu cầu, phải gọi đúng công cụ cần thiết
4. Khi trả lời, phải tuân thủ định dạng (Thought/Action/Observation/Final Answer)
5. Nếu không chắc chắn, hãy yêu cầu thêm thông tin từ khách hàng thay vì bịa dữ liệu.
6. KHÔNG BAO GIỜ tự tạo ticket_id. PHẢI gọi tool `submit_support_ticket` để hệ thống tự sinh.
7. Nếu khách hàng hỏi câu FAQ đơn giản (chính sách bảo hành, đổi trả), trả lời trực tiếp KHÔNG cần gọi tool.

## OPERATIONAL BOUNDARIES
- Không trả lời các câu hỏi ngoài phạm vi sản phẩm/dịch vụ Vingroup.
- Không trả lời các câu hỏi về chính trị, tôn giáo, hoặc các chủ đề nhạy cảm.
- Không trả lời các câu hỏi về các công ty khác ngoài Vingroup.

## OUTPUT CONTRACT
- Khi liệt kê sản phẩm, dùng bảng hoặc danh sách có thứ tự.
- Khi tạo ticket, trả về ticket_id và trạng thái.
- Khi trả lời, luôn tuân thủ định dạng (Thought/Action/Observation/Final Answer).
- luôn trả lời bằng tiếng Việt, trừ khi khách hàng yêu cầu khác.
## RESPONSE FORMAT
- Thought: <suy nghĩ của bạn về yêu cầu của khách hàng>
- Action: <công cụ bạn sẽ gọi, nếu cần>
- Observation: <kết quả trả về từ công cụ, nếu có>
- Final Answer: <câu trả lời cuối cùng cho khách hàng>

"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        """Trả lời một lượt mà không sử dụng bất kỳ tool nào."""
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def _detect_intents(self, user_input: str) -> Dict[str, Any]:
        """Xác định yêu cầu catalog, ticket hoặc FAQ từ câu hỏi."""
        text = user_input.lower()
        is_faq = any(word in text for word in ("bảo hành", "chính sách", "đổi trả"))
        needs_catalog = any(
            word in text
            for word in ("xe điện", "xe vinfast", "vinfast", "vinpearl", "resort", "du lịch")
        ) and any(word in text for word in ("xem", "tìm", "giá", "dưới", "mua"))
        needs_ticket = any(
            word in text
            for word in ("lỗi", "hỏng", "hỗ trợ", "khiếu nại", "phản ánh", "sự cố", "ticket")
        )

        category = "du_lich" if any(
            word in text for word in ("vinpearl", "resort", "du lịch")
        ) else "xe_dien"
        price_match = re.search(r"(\d+)\s*(?:triệu|tr)", text)
        catalog_args: Dict[str, Any] = {"category": category}
        if price_match:
            catalog_args["max_price"] = int(price_match.group(1)) * 1_000_000

        name_match = re.search(
            r"(?:tên tôi là|tôi tên là|tôi là|tên)\s+([A-ZÀ-Ỹ][\wÀ-ỹ]*(?:\s+[A-ZÀ-Ỹ][\wÀ-ỹ]*)*)",
            user_input,
        )
        customer_name = name_match.group(1).strip() if name_match else "Khách hàng ẩn danh"
        priority = "high" if any(
            word in text for word in ("gấp", "khẩn cấp", "nghiêm trọng", "cấp bách")
        ) else "medium"

        return {
            "is_faq": is_faq and not needs_catalog and not needs_ticket,
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "catalog_args": catalog_args,
            "ticket_args": {
                "customer_name": customer_name,
                "issue_description": user_input,
                "priority": priority,
            },
        }

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        intents = self._detect_intents(user_input)
        if intents["is_faq"]:
            answer = (
                "Chính sách bảo hành pin xe điện VinFast có thể lên đến 10 năm "
                "tùy dòng xe và điều kiện áp dụng. Bạn nên liên hệ VinFast để "
                "xác nhận chính sách cụ thể cho xe của mình."
            )
            self.trace.append({"iteration": 1, "thought": "FAQ, không cần gọi tool."})
            return {"answer": answer, "trace": self.trace, "iterations": 1, "status": "completed"}

        iteration = 0
        observations: Dict[str, Any] = {}
        if intents["needs_catalog"]:
            iteration += 1
            args = intents["catalog_args"]
            observations["catalog"] = TOOL_MAP["search_product_catalog"](**args)
            self.trace.append({
                "iteration": iteration,
                "action": {"name": "search_product_catalog", "args": args},
                "observation": observations["catalog"],
            })

        if intents["needs_ticket"]:
            iteration += 1
            args = intents["ticket_args"]
            observations["ticket"] = TOOL_MAP["submit_support_ticket"](**args)
            self.trace.append({
                "iteration": iteration,
                "action": {"name": "submit_support_ticket", "args": args},
                "observation": observations["ticket"],
            })

        if iteration == 0:
            iteration = 1
            answer = "Xin lỗi, tôi chỉ hỗ trợ sản phẩm, dịch vụ và yêu cầu hỗ trợ của Vingroup."
        elif "catalog" in observations and not observations["catalog"]:
            answer = "Rất tiếc, không tìm thấy sản phẩm phù hợp với yêu cầu của bạn."
        elif "ticket" in observations and "catalog" not in observations:
            ticket = observations["ticket"]
            answer = f"Đã tạo ticket {ticket['ticket_id']} cho {ticket['customer_name']}. Trạng thái: {ticket['status']}."
        elif "catalog" in observations:
            products = observations["catalog"]
            names = ", ".join(product["name"] for product in products)
            answer = f"Tìm thấy {len(products)} sản phẩm phù hợp: {names}."
            if "ticket" in observations:
                answer += f" Ticket {observations['ticket']['ticket_id']} đã được tạo."
        else:
            answer = "Đã xử lý yêu cầu của bạn."

        if iteration > self.max_iterations:
            return {
                "answer": "Lỗi: Agent đã vượt quá số bước lặp tối đa.",
                "trace": self.trace,
                "iterations": self.max_iterations,
                "status": "max_iterations_reached",
            }
        return {
            "answer": answer,
            "trace": self.trace,
            "iterations": iteration,
            "status": "completed",
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
