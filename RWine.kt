package smile

import smile.data.DataFrame
import org.apache.commons.csv.CSVFormat
import smile.regression.GradientTreeBoost
import smile.data.formula.Formula
import smile.io.Read
import smile.validation.CrossValidation

fun main() {

    val wine: DataFrame = Read.csv("src/main/resources/winequality-red.csv", CSVFormat.DEFAULT.withFirstRecordAsHeader())
    println(wine)

    val f = Formula.lhs("quality")

    val res = CrossValidation.regression(
        10, f, wine,
        { formula, data -> GradientTreeBoost.fit(formula, data) })
    println(res)
}