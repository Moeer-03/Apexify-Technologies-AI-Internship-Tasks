import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class FAQChatbot:
    def __init__(self):
        """Initialize the chatbot with FAQ database."""
        self.faqs = self.load_faqs()
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            max_features=100
        )
        self.faq_vectors = self.vectorize_faqs()
        self.conversation_history = []

    def load_faqs(self):
        """Load FAQ database - can be extended with external data."""
        faqs = [
            {
                "question": "What are your business hours?",
                "answer": "We are open Monday to Friday from 9 AM to 6 PM, and Saturday from 10 AM to 4 PM. We are closed on Sundays.",
                "keywords": ["hours", "open", "close", "timing"]
            },
            {
                "question": "How do I reset my password?",
                "answer": "To reset your password, click 'Forgot Password' on the login page, enter your email address, and follow the instructions sent to your email.",
                "keywords": ["password", "reset", "forgot", "login"]
            },
            {
                "question": "What payment methods do you accept?",
                "answer": "We accept all major credit cards (Visa, MasterCard, American Express), PayPal, Apple Pay, and bank transfers.",
                "keywords": ["payment", "credit card", "accept", "method"]
            },
            {
                "question": "How long does shipping take?",
                "answer": "Standard shipping takes 5-7 business days. Express shipping takes 2-3 business days. International shipping takes 10-14 business days.",
                "keywords": ["shipping", "delivery", "time", "days"]
            },
            {
                "question": "What is your return policy?",
                "answer": "We offer a 30-day money-back guarantee. Items must be in original condition with all packaging and accessories included.",
                "keywords": ["return", "refund", "exchange", "policy"]
            },
            {
                "question": "How do I contact customer support?",
                "answer": "You can reach us via email at support@company.com, phone at 1-800-123-4567, or through our live chat on the website.",
                "keywords": ["contact", "support", "help", "email", "phone"]
            },
            {
                "question": "Do you offer discounts for bulk purchases?",
                "answer": "Yes! We offer volume discounts starting at 10+ items. Contact our sales team for a custom quote.",
                "keywords": ["discount", "bulk", "price", "wholesale"]
            },
            {
                "question": "Is my personal data secure?",
                "answer": "Yes, we use 256-bit SSL encryption and comply with GDPR standards. Your data is never shared with third parties.",
                "keywords": ["security", "privacy", "data", "safe", "encrypt"]
            },
            {
                "question": "Can I track my order?",
                "answer": "Yes, you'll receive a tracking number via email once your order ships. You can track it on our website.",
                "keywords": ["track", "order", "status", "shipping", "number"]
            },
            {
                "question": "Do you have a physical store?",
                "answer": "We operate online only, but we ship worldwide. You can visit our website or contact us for store locator information.",
                "keywords": ["store", "location", "physical", "visit"]
            }
        ]
        return faqs

    def preprocess_text(self, text):
        """Preprocess text: lowercase, remove special characters, extra spaces."""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def vectorize_faqs(self):
        """Convert FAQ questions to TF-IDF vectors."""
        questions = [faq["question"] for faq in self.faqs]
        preprocessed_questions = [self.preprocess_text(q) for q in questions]
        vectors = self.vectorizer.fit_transform(preprocessed_questions)
        return vectors

    def find_best_match(self, user_question, threshold=0.3):
        """
        Find the best matching FAQ using cosine similarity.
        Returns the FAQ and similarity score.
        """
        preprocessed_question = self.preprocess_text(user_question)
        user_vector = self.vectorizer.transform([preprocessed_question])

        # Calculate cosine similarity
        similarities = cosine_similarity(user_vector, self.faq_vectors)[0]
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        # Check if similarity meets threshold
        if best_score < threshold:
            return None, best_score

        return self.faqs[best_idx], best_score

    def get_response(self, user_question):
        """Get chatbot response for user question."""
        if not user_question.strip():
            return "Please enter a question."

        faq, score = self.find_best_match(user_question)

        if faq is None:
            return (
                "I'm not sure about that. Try asking about: business hours, "
                "password reset, payments, shipping, returns, contact info, "
                "discounts, security, tracking, or store locations."
            )

        response = f"Q: {faq['question']}\n\nA: {faq['answer']}\n\n"
        response += f"(Confidence: {score*100:.1f}%)"

        return response

    def display_welcome(self):
        """Display welcome message."""
        print("\n" + "="*60)
        print("FAQ CHATBOT")
        print("="*60)
        print("Welcome! I'm here to help answer your questions.")
        print("Type 'list' to see available topics")
        print("Type 'quit' to exit")
        print("="*60 + "\n")

    def display_topics(self):
        """Display available FAQ topics."""
        print("\nAvailable Topics:")
        print("-" * 60)
        for i, faq in enumerate(self.faqs, 1):
            print(f"{i}. {faq['question']}")
        print("-" * 60 + "\n")

    def run(self):
        """Main chatbot loop."""
        self.display_welcome()

        while True:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                print("\nThank you for using FAQ Chatbot. Goodbye!")
                break

            if user_input.lower() == 'list':
                self.display_topics()
                continue

            # Get and display response
            response = self.get_response(user_input)
            print(f"\nBot: {response}\n")

            # Store conversation
            self.conversation_history.append({
                "user": user_input,
                "bot": response
            })


class FAQWebChatbot(FAQChatbot):
    """Extended chatbot with additional features for web-like interaction."""

    def __init__(self):
        super().__init__()
        self.session_id = None

    def get_quick_replies(self, user_question):
        """Generate quick reply suggestions based on user question."""
        faq, score = self.find_best_match(user_question)

        if faq:
            # Find related FAQs
            related = []
            for i, other_faq in enumerate(self.faqs):
                if other_faq != faq:
                    similarity = cosine_similarity(
                        self.vectorizer.transform([self.preprocess_text(other_faq["question"])]),
                        self.vectorizer.transform([self.preprocess_text(faq["question"])])
                    )[0][0]
                    if similarity > 0.2:
                        related.append(other_faq["question"])

            return related[:3]  # Return top 3 related questions
        return []

    def run_web_mode(self):
        """Run in web-like mode with quick replies and history."""
        self.display_welcome()

        while True:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                self.display_conversation_summary()
                print("\nThank you for using FAQ Chatbot. Goodbye!")
                break

            if user_input.lower() == 'list':
                self.display_topics()
                continue

            if user_input.lower() == 'history':
                self.display_conversation_summary()
                continue

            # Get response
            response = self.get_response(user_input)
            print(f"\nBot: {response}\n")

            # Get and show quick replies
            quick_replies = self.get_quick_replies(user_input)
            if quick_replies:
                print("You might also want to ask:")
                for i, reply in enumerate(quick_replies, 1):
                    print(f"  {i}. {reply}")
                print()

            self.conversation_history.append({
                "user": user_input,
                "bot": response
            })

    def display_conversation_summary(self):
        """Display conversation history."""
        if not self.conversation_history:
            print("\nNo conversation history yet.\n")
            return

        print("\n" + "="*60)
        print("CONVERSATION HISTORY")
        print("="*60)
        for i, exchange in enumerate(self.conversation_history, 1):
            print(f"\n[{i}] You: {exchange['user']}")
            print(f"    Bot: {exchange['bot'][:100]}...")
        print("\n" + "="*60 + "\n")


def main():
    """Main entry point."""
    print("\nSelect mode:")
    print("1. Basic Chatbot")
    print("2. Web-like Chatbot (with suggestions and history)")

    choice = input("Enter 1 or 2: ").strip()

    if choice == '2':
        chatbot = FAQWebChatbot()
        chatbot.run_web_mode()
    else:
        chatbot = FAQChatbot()
        chatbot.run()


if __name__ == "__main__":
    main()
