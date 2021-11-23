import numpy as np

def nn_matrix_to_latex(matrix):
    maxNWeights = 0
    for arr in matrix:
        maxNWeights = max(maxNWeights, arr.shape[1])

    nodeNames = [str(i) for i in range(maxNWeights)]
    layerNames = [str(i) for i in range(1, len(matrix)+1)]

    rowDelim = '&'
    rowSpaceDelim = ' ' + rowDelim + ' %s'
    rowSuffix = '\\\\\n'
    hline = '\\hline\n'

    latexTabular = "\\begin{tabular}{" + ''.join([' c' for _ in range(maxNWeights+2)])[1:] + "}\n"
    latexTabular += hline + 'Layer & Node & Bias' +\
                   ''.join([rowSpaceDelim % "Weight " + str(i) for i in range(maxNWeights-1)]) + rowSuffix +hline
    for lWeights, lName in zip(matrix, layerNames):
        for nWeights, nName in zip(lWeights, nodeNames):
            additionalAnds = [rowDelim for _ in range(maxNWeights - len(nWeights))]
            weightStrs = [rowSpaceDelim % ('%.3f' % w) for w in nWeights]
            toAppend = ''.join(
                [lName, rowSpaceDelim % nName] +
                weightStrs +
                additionalAnds + [rowSuffix]
            )
            latexTabular += toAppend
    latexTabular += hline + "\\end{tabular}"
    return latexTabular