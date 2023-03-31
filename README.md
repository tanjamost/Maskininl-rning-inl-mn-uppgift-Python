Iris Dataset Classification Model med FastAPI

Det här är ett enkelt projekt som visar hur man tränar en classification model på Iris dataset och tjänar den som ett webb-API med FastAPI.

Modellen tränas med den random forest algorithm från scikit-learn, och API:n implementeras med hjälp av FastAPI, ett modernt och snabbt webbramverk för att bygga API:er med Python.

Användande

För att använda API:t, skicka en POST-begäran till /predict endpoint med en JSON-payload innehåller värden för de fyra input features i Iris-dataset. Här är ett exempel på JSON-payload:

{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

Denna JSON-payload motsvarar input features för en enskild datapunkt i Iris-dataset. När du skickar den här JSON-payload till /predict endpoint för API:t kommer den tränade classification model att använda input features för att förutsäga iris arten för den givna data point.

Här är ett exempel på utdata du kan förvänta dig från API:n:

{
    "predicted_class": "setosa"
}

Denna JSON-output innehåller de predicted iris arterna för de givna input features, som i det här fallet är "setosa".