#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test OpenAI API connection
"""

import os
import openai


def test_openai_api():
    """Test OpenAI API connection"""
    print("Testing OpenAI API connection...")

    # Check API key
    api_key = "sk-proj-xxxx"
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        return False

    print(f"✅ API key set: {api_key[:10]}...")

    try:
        # Setup client
        client = openai.OpenAI(api_key=api_key)

        # Test simple request
        print("Testing API connection...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, this is a test message."}],
            max_tokens=10,
        )

        print("✅ API connection successful!")
        print(f"Response: {response.choices[0].message.content}")
        return True

    except openai.RateLimitError as e:
        print("❌ Rate limit error:")
        print(f"Error message: {e}")
        print("\nSolutions:")
        print("1. Check your OpenAI account balance")
        print("2. Upgrade to a paid plan")
        print("3. Add payment method")
        return False

    except openai.AuthenticationError as e:
        print("❌ Authentication error:")
        print(f"Error message: {e}")
        print("\nSolutions:")
        print("1. Check if your API key is correct")
        print("2. Make sure the API key doesn't have extra spaces")
        return False

    except Exception as e:
        print(f"❌ Other error: {e}")
        return False


def test_embedding_api():
    """Test Embedding API"""
    print("\nTesting Embedding API...")

    try:
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        response = client.embeddings.create(
            model="text-embedding-3-small", input="This is a test sentence."
        )

        print("✅ Embedding API connection successful!")
        print(f"Vector dimension: {len(response.data[0].embedding)}")
        return True

    except Exception as e:
        print(f"❌ Embedding API error: {e}")
        return False


def main():
    """Main function"""
    print("OpenAI API Connection Test")
    print("=" * 40)

    # Test chat API
    chat_ok = test_openai_api()

    # Test embedding API
    embedding_ok = test_embedding_api()

    print("\n" + "=" * 40)
    if chat_ok and embedding_ok:
        print("✅ All API tests passed! You can run the complete RAG system.")
    else:
        print("❌ Some API tests failed, please check quota and settings.")
    print("=" * 40)


if __name__ == "__main__":
    main()
