
def str_table(tab, lig_names):
    l_first_col = max(len(a) for a in lig_names)
    l_col = max(len(str(a)) for ligne in tab for a in ligne)
    c = len(tab[0])
    res = "╔" + "═" * (l_first_col + 2) + ("╤══" + "═" * l_col) * c + "╗\n" + "".join(
        ("║{:^" + str(l_first_col + 2) + "}" + "".join("│{:^" + str(l_col + 2) + "}" for _ in range(c)) + "║\n").format(
            *[lig_names[ligne]] + [str(a) for a in tab[ligne]]) for ligne in range(len(tab))) + "╚" + "═" * (
                  l_first_col + 2) + ("╧══" + "═" * l_col) * c + "╝"
    return res
