from llm_kee.models import LearningSignal, SignalType, UserFeedback


class FeedbackInterpreter:
    def interpret(self, feedback: UserFeedback) -> LearningSignal:
        target = feedback.target_type
        action = feedback.feedback_type
        summary = f"{action} feedback for {target}"
        if feedback.comment:
            summary = f"{summary}: {feedback.comment}"
        return LearningSignal(
            signal_type=SignalType.FEEDBACK,
            source_id=feedback.id,
            summary=summary,
            payload=feedback.model_dump(mode="json"),
            priority=8 if action in {"correction", "conflict"} else 5,
        )
