from IPython.display import Math,display

def display_latex(string_var):
  string_var = str(string_var)
  new_str = ""
  for c in string_var:
    if c =='\\':
      new_str = new_str + c
    else:
      pass
    
    new_str = new_str + c

  print(string_var)
  display(Math("\large " + string_var))

display(Math( "\large (n - f + 1 ) " ))

display_latex("(n - f * 3")

display_latex(" * x X \dot \times")

display(Math('\\backslash \\times '))

display_latex("\\dots \\cdots")

display_latex("x \\times \\cdot \\odot \\dot \\ Y \\dot{Y} . y")

"""latex experiment
x        y \\
xy\\
x y \\
x \ y \\
x + y \\
X \\
\odot \\
examples \times rows \times columns \times channels \\
examples \ x \ rows\  X\  columns \ x \  channels \\
examples \dot rows \dot columns \\
examples \cdot rows \cdot columns \\
\\bullet
"""