from django.urls import path 
from .views import ProductTrainPipeline ,ProductSemanticSearchView


urlpatterns = [
    path('product-data-train' ,ProductTrainPipeline.as_view()),
    path('semantic-search' ,ProductSemanticSearchView.as_view() )
]



