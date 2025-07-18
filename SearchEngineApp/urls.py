from django.urls import path 
from .views import SemanticSearchView


urlpatterns = [
    path('semantic-search' ,SemanticSearchView.as_view() )
]


