from app.memory.memory_manager import MemoryManager
from app.llm.gguf_model import GGUFModel


def run_test():
    #  Step 1: Initialize LLM
    llm = GGUFModel()

    #  Step 2: Initialize Memory Manager
    manager = MemoryManager(llm)

    session_id = "test_user"

    print("\n=== START MEMORY TEST ===\n")

    #  Step 3: Simulate conversation
    for i in range(12):
        user_msg = f"What is concept {i}?"
        assistant_msg = f"Concept {i} is explained here."

        result = manager.add_interaction(
            session_id,
            user_msg,
            assistant_msg
        )

        print(f"\nInteraction {i+1}")
        print("User:", user_msg)
        print("Assistant:", assistant_msg)
        print("Result:", result)

        # Show current memory
        history = manager.get_history(session_id)
        print("\nCurrent Memory:")
        for msg in history:
            print(msg)

    print("\n=== TEST COMPLETE ===\n")


if __name__ == "__main__":
    run_test()