(TeX-add-style-hook "rules"
 (lambda ()
    (LaTeX-add-labels
     "first-iteration"
     "following-iterations")
    (TeX-run-style-hooks
     "latex2e"
     "art10"
     "article")))

