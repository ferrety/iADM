(TeX-add-style-hook "Testing of ideas"
 (lambda ()
    (LaTeX-add-environments
     "Shaded")
    (LaTeX-add-labels
     "first-iteration"
     "following-iterations")
    (TeX-add-symbols
     '("NormalTok" 1)
     '("ErrorTok" 1)
     '("RegionMarkerTok" 1)
     '("FunctionTok" 1)
     '("AlertTok" 1)
     '("OtherTok" 1)
     '("CommentTok" 1)
     '("StringTok" 1)
     '("CharTok" 1)
     '("FloatTok" 1)
     '("BaseNTok" 1)
     '("DecValTok" 1)
     '("DataTypeTok" 1)
     '("KeywordTok" 1)
     "tightlist"
     "br"
     "gt"
     "lt")
    (TeX-run-style-hooks
     "booktabs"
     "longtable"
     "hyperref"
     "grffile"
     "fancyvrb"
     "inputenc"
     "utf8x"
     "ucs"
     "mathletters"
     "eurosym"
     "amssymb"
     "amsmath"
     "geometry"
     "enumerate"
     "color"
     "adjustbox"
     "graphicx")))

