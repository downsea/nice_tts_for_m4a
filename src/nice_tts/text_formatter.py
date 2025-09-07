"""
Text formatting and post-processing utilities for Nice-TTS.
Improves readability of transcribed text by adding punctuation and structure.
"""

import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class TextFormatter:
    """Formats transcribed text for better readability."""
    
    def __init__(self, language: str = "zh"):
        self.language = language
        self.punctuation_rules = self._get_punctuation_rules()
        
    def _get_punctuation_rules(self) -> Dict[str, Any]:
        """Get language-specific punctuation rules."""
        rules = {
            "zh": {
                "sentence_enders": ["吗", "呢", "吧", "啊", "呀", "嘛", "的", "了", "呢"],
                "pause_words": ["然后", "接着", "但是", "不过", "可是", "而且", "另外", "其次"],
                "question_particles": ["吗", "呢", "吧", "啊"],
                "exclamation_words": ["真的", "太好了", "太棒了", "天啊", "哇"],
                "min_sentence_length": 8,
                "max_sentence_length": 30,
            },
            "en": {
                "sentence_enders": [".", "!", "?"],
                "pause_words": ["however", "but", "and", "then", "also", "moreover", "furthermore"],
                "question_particles": ["what", "how", "when", "where", "why", "who"],
                "exclamation_words": ["amazing", "great", "wonderful", "awesome"],
                "min_sentence_length": 10,
                "max_sentence_length": 25,
            }
        }
        return rules.get(self.language, rules["en"])
    
    def format_text(self, text: str, add_punctuation: bool = True, 
                   segment_sentences: bool = True, max_line_length: int = 80) -> str:
        """
        Format transcribed text for better readability.
        
        Args:
            text: Raw transcribed text
            add_punctuation: Add missing punctuation
            segment_sentences: Break into logical sentences
            max_line_length: Maximum characters per line
            
        Returns:
            Formatted text with improved readability
        """
        if not text or not text.strip():
            return text
            
        # Clean up the text first
        formatted_text = self._clean_text(text)
        
        if add_punctuation:
            formatted_text = self._add_punctuation(formatted_text)
            
        if segment_sentences:
            formatted_text = self._segment_sentences(formatted_text)
            
        # Apply line wrapping
        if max_line_length > 0:
            formatted_text = self._wrap_lines(formatted_text, max_line_length)
            
        return formatted_text.strip()
    
    def _clean_text(self, text: str) -> str:
        """Clean up basic text issues."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove obvious artifacts
        text = re.sub(r'[\[\]{}()]*', '', text)
        
        # Fix common transcription errors
        if self.language == "zh":
            # Fix common Chinese transcription issues
            text = text.replace(" ", "")  # Remove spaces in Chinese text
            text = re.sub(r'([。！？，])\1+', r'\1', text)  # Remove duplicate punctuation
        
        return text
    
    def _add_punctuation(self, text: str) -> str:
        """Add missing punctuation based on context."""
        rules = self.punctuation_rules
        
        if self.language == "zh":
            return self._add_chinese_punctuation(text, rules)
        else:
            return self._add_english_punctuation(text, rules)
    
    def _add_chinese_punctuation(self, text: str, rules: Dict[str, Any]) -> str:
        """Add punctuation to Chinese text."""
        sentences = []
        current_sentence = ""
        
        words = list(text)  # Chinese characters
        
        for i, char in enumerate(words):
            current_sentence += char
            
            # Check for natural sentence endings
            if char in rules["sentence_enders"] and i > 0:
                # Check if this looks like a question
                if char in rules["question_particles"] and self._looks_like_question(current_sentence):
                    sentences.append(current_sentence + "？")
                    current_sentence = ""
                # Check if this looks like exclamation
                elif self._looks_like_exclamation(current_sentence):
                    sentences.append(current_sentence + "！")
                    current_sentence = ""
                # Regular statement
                elif len(current_sentence) >= rules["min_sentence_length"]:
                    sentences.append(current_sentence + "。")
                    current_sentence = ""
            
            # Check for pause words that need commas
            elif char in ["的", "了", "吗", "呢"] and len(current_sentence) > 15:
                # Only add comma if it makes sense contextually
                if i + 1 < len(words) and words[i + 1] not in ["，", "。", "！", "？"]:
                    sentences.append(current_sentence + "，")
                    current_sentence = ""
            
            # Look ahead for natural break points
            elif len(current_sentence) >= rules["max_sentence_length"]:
                # Try to find a better break point
                best_break = len(current_sentence)
                for j in range(len(current_sentence) - 5, len(current_sentence)):
                    if current_sentence[j] in ["，", "的", "了", "吗", "呢"]:
                        best_break = j + 1
                        break
                
                if best_break < len(current_sentence):
                    sentences.append(current_sentence[:best_break] + "。")
                    current_sentence = current_sentence[best_break:]
                else:
                    sentences.append(current_sentence + "。")
                    current_sentence = ""
        
        # Add remaining text
        if current_sentence:
            if len(current_sentence) > 5:
                # Try to determine appropriate ending punctuation
                if self._looks_like_question(current_sentence):
                    sentences.append(current_sentence + "？")
                elif self._looks_like_exclamation(current_sentence):
                    sentences.append(current_sentence + "！")
                else:
                    sentences.append(current_sentence + "。")
            else:
                sentences.append(current_sentence)
        
        return "".join(sentences)
    
    def _add_english_punctuation(self, text: str, rules: Dict[str, Any]) -> str:
        """Add punctuation to English text."""
        # Simple approach for English - capitalize first letter and add periods
        sentences = []
        current_sentence = ""
        
        words = text.split()
        
        for i, word in enumerate(words):
            current_sentence += word + " "
            
            # Check for natural endings
            if word.lower() in ["the", "and", "but", "then"] and len(current_sentence.split()) > 8:
                sentences.append(current_sentence.strip() + ",")
                current_sentence = ""
            elif len(current_sentence.split()) >= rules["max_sentence_length"]:
                sentences.append(current_sentence.strip().capitalize() + ".")
                current_sentence = ""
        
        # Add remaining text
        if current_sentence.strip():
            sentences.append(current_sentence.strip().capitalize() + ".")
        
        return " ".join(sentences)
    
    def _looks_like_question(self, sentence: str) -> bool:
        """Determine if a sentence is likely a question."""
        if self.language == "zh":
            # Chinese question indicators
            question_indicators = ["吗", "呢", "什么", "怎么", "为什么", "哪里", "谁", "几"]
            return any(indicator in sentence for indicator in question_indicators)
        else:
            # English question indicators
            question_starters = ["what", "how", "when", "where", "why", "who", "which"]
            return any(sentence.lower().startswith(starter) for starter in question_starters)
    
    def _looks_like_exclamation(self, sentence: str) -> bool:
        """Determine if a sentence is likely an exclamation."""
        if self.language == "zh":
            exclamation_words = ["真的", "太好了", "太棒了", "天啊", "哇", "唉呀", "好厉害"]
            return any(word in sentence for word in exclamation_words)
        else:
            exclamation_words = ["amazing", "great", "wonderful", "awesome", "fantastic"]
            return any(word in sentence.lower() for word in exclamation_words)
    
    def _segment_sentences(self, text: str) -> str:
        """Break text into logical paragraphs and sentences."""
        if not text:
            return text
            
        # Split by sentence endings
        if self.language == "zh":
            sentences = re.split(r'([。！？])', text)
            # Rejoin punctuation with preceding text
            formatted_sentences = []
            for i in range(0, len(sentences)-1, 2):
                if i+1 < len(sentences):
                    formatted_sentences.append(sentences[i] + sentences[i+1])
                else:
                    formatted_sentences.append(sentences[i])
            if len(sentences) % 2 == 1:
                formatted_sentences.append(sentences[-1])
        else:
            sentences = re.split(r'([.!?])', text)
            formatted_sentences = []
            for i in range(0, len(sentences)-1, 2):
                if i+1 < len(sentences):
                    formatted_sentences.append(sentences[i].strip() + sentences[i+1])
                else:
                    formatted_sentences.append(sentences[i].strip())
            if len(sentences) % 2 == 1:
                formatted_sentences.append(sentences[-1].strip())
        
        # Group sentences into paragraphs (every 3-4 sentences)
        paragraphs = []
        current_paragraph = []
        
        for sentence in formatted_sentences:
            if sentence.strip():
                current_paragraph.append(sentence.strip())
                
                # Start new paragraph after 3-4 sentences
                if len(current_paragraph) >= 4:
                    paragraphs.append(" ".join(current_paragraph))
                    current_paragraph = []
        
        # Add remaining sentences
        if current_paragraph:
            paragraphs.append(" ".join(current_paragraph))
        
        return "\n\n".join(paragraphs)
    
    def _wrap_lines(self, text: str, max_length: int) -> str:
        """Wrap long lines for better readability."""
        if not text or max_length <= 0:
            return text
            
        lines = text.split('\n')
        wrapped_lines = []
        
        for line in lines:
            if len(line) <= max_length:
                wrapped_lines.append(line)
            else:
                # Wrap long lines
                words = line.split()
                current_line = ""
                
                for word in words:
                    if len(current_line + word) <= max_length:
                        current_line += word + " "
                    else:
                        if current_line:
                            wrapped_lines.append(current_line.strip())
                        current_line = word + " "
                
                if current_line:
                    wrapped_lines.append(current_line.strip())
        
        return "\n".join(wrapped_lines)
    
    def format_segments(self, segments: List[Dict[str, Any]], 
                       include_timestamps: bool = True) -> str:
        """
        Format transcription segments with timestamps.
        
        Args:
            segments: List of transcription segments
            include_timestamps: Whether to include timestamps
            
        Returns:
            Formatted segments text
        """
        if not segments:
            return ""
            
        formatted_segments = []
        
        for i, segment in enumerate(segments, 1):
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            text = segment.get("text", "").strip()
            
            if not text:
                continue
                
            # Format individual segment text
            formatted_text = self.format_text(text, add_punctuation=True, 
                                            segment_sentences=False, max_line_length=0)
            
            if include_timestamps:
                formatted_segments.append(f"[{start:.1f}s - {end:.1f}s] {formatted_text}")
            else:
                formatted_segments.append(f"{formatted_text}")
        
        return "\n".join(formatted_segments)


def format_transcription_readable(text: str, language: str = "zh", 
                                add_punctuation: bool = True,
                                max_line_length: int = 80) -> str:
    """
    Convenience function to format transcription text.
    
    Args:
        text: Raw transcription text
        language: Language code ("zh" or "en")
        add_punctuation: Whether to add punctuation
        max_line_length: Maximum line length for wrapping
        
    Returns:
        Formatted text with improved readability
    """
    formatter = TextFormatter(language=language)
    return formatter.format_text(text, add_punctuation=add_punctuation, 
                               segment_sentences=True, max_line_length=max_line_length)


# Global formatter instance
formatter = TextFormatter()