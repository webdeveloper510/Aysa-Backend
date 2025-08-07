from django.urls import path 
from .views import *


urlpatterns = [
    path('product-data-train' ,ProductTrainPipeline.as_view()),
    path('tax-data-train' ,TaxTrainPipeline.as_view()),
    path('ceo-worker-data-train' ,CEOWorkerTrainPipeline.as_view()),

    path('product-semantic-search' ,ProductSemanticSearchView.as_view()),
    path('tax-semantic-search' ,TaxSemanticSearchView.as_view()),
    path('ceo-worker-semantic-search' ,CEOWorkerSemanticSearchView.as_view()),

    # GET ALL product margin data 
    path("get-profit-margin-data", GetProfitMarginData.as_view())
]



