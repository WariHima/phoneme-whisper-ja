"""
以下のコードは VOICEVOX のソースコードからお借りし最低限の改造を行ったもの。
https://github.com/VOICEVOX/voicevox_engine/blob/master/voicevox_engine/tts_pipeline/mora_list.py
"""

"""
以下のモーラ対応表は OpenJTalk のソースコードから取得し、
カタカナ表記とモーラが一対一対応するように改造した。
ライセンス表記：
-----------------------------------------------------------------
          The Japanese TTS System "Open JTalk"
          developed by HTS Working Group
          http://open-jtalk.sourceforge.net/
-----------------------------------------------------------------

 Copyright (c) 2008-2014  Nagoya Institute of Technology
                          Department of Computer Science

All rights reserved.

Redistribution and use in source and binary forms, with or
without modification, are permitted provided that the following
conditions are met:

- Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the following
  disclaimer in the documentation and/or other materials provided
  with the distribution.
- Neither the name of the HTS working group nor the names of its
  contributors may be used to endorse or promote products derived
  from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""


# (カタカナ, 子音, 母音)の順。子音がない場合は None を入れる。
# 但し「ン」と「ッ」は母音のみという扱いで、「ン」は「N」、「ッ」は「q」とする。
# （元々「ッ」は「cl」）
# また pyopenjtalk-plus で追加されたモーラと、エイリアスとして「ヂャ」「ヂュ」「ヂョ」「ヂェ」を独自に追加している
# 以下に定義されているモーラは、「ヂャ」「ヂュ」「ヂョ」「ヂェ」を除き全て pyopenjtalk-plus (に内蔵されている OpenJTalk) に定義されているもの
# ref: https://github.com/VOICEVOX/voicevox_engine/pull/1473
__MORA_LIST_MINIMUM: list[tuple[str, str | None, str]] = [
    ("ヴォ", "v", "o"),
    ("ヴェ", "v", "e"),
    ("ヴィ", "v", "i"),
    ("ヴァ", "v", "a"),
    ("ヴ", "v", "u"),
    ("ン", "", "N"),
    ("ワ", "w", "a"),
    ("ロ", "r", "o"),
    ("レ", "r", "e"),
    ("ル", "r", "u"),
    ("リョ", "ry", "o"),
    ("リュ", "ry", "u"),
    ("リャ", "ry", "a"),
    ("リェ", "ry", "e"),
    ("リ", "r", "i"),
    ("ラ", "r", "a"),
    ("ヨ", "y", "o"),
    ("ユ", "y", "u"),
    ("ヤ", "y", "a"),
    ("モ", "m", "o"),
    ("メ", "m", "e"),
    ("ム", "m", "u"),
    ("ミョ", "my", "o"),
    ("ミュ", "my", "u"),
    ("ミャ", "my", "a"),
    ("ミェ", "my", "e"),
    ("ミ", "m", "i"),
    ("マ", "m", "a"),
    ("ポ", "p", "o"),
    ("ボ", "b", "o"),
    ("ホ", "h", "o"),
    ("ペ", "p", "e"),
    ("ベ", "b", "e"),
    ("ヘ", "h", "e"),
    ("プ", "p", "u"),
    ("ブ", "b", "u"),
    ("フュ", "fy", "u"),  # pyopenjtalk-plus で追加されたモーラ
    ("フォ", "f", "o"),
    ("フェ", "f", "e"),
    ("フィ", "f", "i"),
    ("ファ", "f", "a"),
    ("フ", "f", "u"),
    ("ピョ", "py", "o"),
    ("ピュ", "py", "u"),
    ("ピャ", "py", "a"),
    ("ピェ", "py", "e"),
    ("ピ", "p", "i"),
    ("ビョ", "by", "o"),
    ("ビュ", "by", "u"),
    ("ビャ", "by", "a"),
    ("ビェ", "by", "e"),
    ("ビ", "b", "i"),
    ("ヒョ", "hy", "o"),
    ("ヒュ", "hy", "u"),
    ("ヒャ", "hy", "a"),
    ("ヒェ", "hy", "e"),
    ("ヒ", "h", "i"),
    ("パ", "p", "a"),
    ("バ", "b", "a"),
    ("ハ", "h", "a"),
    ("ノ", "n", "o"),
    ("ネ", "n", "e"),
    ("ヌ", "n", "u"),
    ("ニョ", "ny", "o"),
    ("ニュ", "ny", "u"),
    ("ニャ", "ny", "a"),
    ("ニェ", "ny", "e"),
    ("ニ", "n", "i"),
    ("ナ", "n", "a"),
    ("ドゥ", "d", "u"),
    ("ド", "d", "o"),
    ("トゥ", "t", "u"),
    ("ト", "t", "o"),
    ("デョ", "dy", "o"),
    ("デュ", "dy", "u"),
    ("デャ", "dy", "a"),
    ("デェ", "dy", "e"),  # pyopenjtalk-plus で追加されたモーラ
    ("ディ", "d", "i"),
    ("デ", "d", "e"),
    ("テョ", "ty", "o"),
    ("テュ", "ty", "u"),
    ("テャ", "ty", "a"),
    ("ティ", "t", "i"),
    ("テ", "t", "e"),
    ("ツォ", "ts", "o"),
    ("ツェ", "ts", "e"),
    ("ツィ", "ts", "i"),
    ("ツァ", "ts", "a"),
    ("ツ", "ts", "u"),
    ("ッ", "", "cl"),  # 「cl」から「q」に変更
    ("チョ", "ch", "o"),
    ("チュ", "ch", "u"),
    ("チャ", "ch", "a"),
    ("チェ", "ch", "e"),
    ("チ", "ch", "i"),
    ("ダ", "d", "a"),
    ("タ", "t", "a"),
    ("ゾ", "z", "o"),
    ("ソ", "s", "o"),
    ("ゼ", "z", "e"),
    ("セ", "s", "e"),
    ("ズィ", "z", "i"),
    ("ズ", "z", "u"),
    ("スィ", "s", "i"),
    ("ス", "s", "u"),
    ("ジョ", "j", "o"),
    ("ジュ", "j", "u"),
    ("ジャ", "j", "a"),
    ("ジェ", "j", "e"),
    ("ジ", "j", "i"),
    ("ショ", "sh", "o"),
    ("シュ", "sh", "u"),
    ("シャ", "sh", "a"),
    ("シェ", "sh", "e"),
    ("シ", "sh", "i"),
    ("ザ", "z", "a"),
    ("サ", "s", "a"),
    ("ゴ", "g", "o"),
    ("コ", "k", "o"),
    ("ゲ", "g", "e"),
    ("ケ", "k", "e"),
    ("グヮ", "gw", "a"),  # pyopenjtalk-plus で追加されたモーラ
    ("グォ", "gw", "o"),  # pyopenjtalk-plus で追加されたモーラ
    ("グェ", "gw", "e"),  # pyopenjtalk-plus で追加されたモーラ
    ("グゥ", "gw", "u"),  # pyopenjtalk-plus で追加されたモーラ
    ("グィ", "gw", "i"),  # pyopenjtalk-plus で追加されたモーラ
    ("グ", "g", "u"),
    ("クヮ", "kw", "a"),  # pyopenjtalk-plus で追加されたモーラ
    ("クォ", "kw", "o"),  # pyopenjtalk-plus で追加されたモーラ
    ("クェ", "kw", "e"),  # pyopenjtalk-plus で追加されたモーラ
    ("クゥ", "kw", "u"),  # pyopenjtalk-plus で追加されたモーラ
    ("クィ", "kw", "i"),  # pyopenjtalk-plus で追加されたモーラ
    ("ク", "k", "u"),
    ("ギョ", "gy", "o"),
    ("ギュ", "gy", "u"),
    ("ギャ", "gy", "a"),
    ("ギェ", "gy", "e"),
    ("ギ", "g", "i"),
    ("キョ", "ky", "o"),
    ("キュ", "ky", "u"),
    ("キャ", "ky", "a"),
    ("キェ", "ky", "e"),
    ("キ", "k", "i"),
    ("ガ", "g", "a"),
    ("カ", "k", "a"),
    ("オ", "", "o"),
    ("エ", "", "e"),
    ("ウォ", "w", "o"),
    ("ウェ", "w", "e"),
    ("ウィ", "w", "i"),
    ("ウ", "", "u"),
    ("イェ", "y", "e"),
    ("イ", "", "i"),
    ("ア", "", "a"),
]

# __MORA_LIST_MINIMUM と同じ子音＋母音の組に対応するエイリアス
# 例えば "ズ" と "ヅ" はどちらとも ("z", "u") に対応する
__MORA_LIST_ADDITIONAL: list[tuple[str, str | None, str]] = [
    ("ヴョ", "by", "o"),
    ("ヴュ", "by", "u"),
    ("ヴャ", "by", "a"),
    ("ヲ", "", "o"),
    ("ヱ", "", "e"),
    ("ヰ", "", "i"),
    ("ヮ", "w", "a"),
    ("ョ", "y", "o"),
    ("ュ", "y", "u"),
    ("ヅ", "z", "u"),
    ("ヂョ", "j", "o"),  # pyopenjtalk-plus には存在しないエイリアス
    ("ヂュ", "j", "u"),  # pyopenjtalk-plus には存在しないエイリアス
    ("ヂャ", "j", "a"),  # pyopenjtalk-plus には存在しないエイリアス
    ("ヂェ", "j", "e"),  # pyopenjtalk-plus には存在しないエイリアス
    ("ヂ", "j", "i"),
    ("シィ", "s", "i"),  # pyopenjtalk-plus で追加されたモーラ
    ("グァ", "gw", "a"),  # pyopenjtalk-plus で追加されたモーラ
    ("クァ", "kw", "a"),  # pyopenjtalk-plus で追加されたモーラ
    ("ヶ", "k", "e"),
    ("ャ", "y", "a"),
    ("ォ", "", "o"),
    ("ェ", "", "e"),
    ("ゥ", "", "u"),
    ("ィ", "", "i"),
    ("ァ", "", "a"),
]

# モーラの音素表記とカタカナの対応表
# 例: "vo" -> "ヴォ", "a" -> "ア"
MORA_PHONEMES_TO_MORA_KATA: dict[str, str] = {
    (consonant or "") + vowel: kana for [kana, consonant, vowel] in __MORA_LIST_MINIMUM
}

# モーラのカタカナ表記と音素の対応表
# 例: "ヴォ" -> ("v", "o"), "ア" -> (None, "a")
MORA_KATA_TO_MORA_PHONEMES: dict[str, tuple[str | None, str]] = {
    kana: (consonant, vowel)
    for [kana, consonant, vowel] in __MORA_LIST_MINIMUM + __MORA_LIST_ADDITIONAL
}

# 子音の集合
CONSONANTS = set(
    [
        consonant
        for consonant, _ in MORA_KATA_TO_MORA_PHONEMES.values()
        if consonant is not None
    ]
)

# 母音の集合 (便宜上「ん」を含める)
VOWELS = {"a", "i", "u", "e", "o", "N"}
