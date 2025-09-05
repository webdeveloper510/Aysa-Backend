from django.urls import path 
from .views import *


urlpatterns = [
    path('product-data-train' ,ProductTrainPipeline.as_view()),
    path('tax-data-train' ,TaxDataTrainPipeline.as_view()),
    path('ceo-worker-data-train' ,CEOWorkerTrainPipeline.as_view()),

    path('product-semantic-search' ,ProductSemanticSearchView.as_view()),
    path('tax-semantic-search' ,TaxSemanticSearchView.as_view()),
    path('ceo-worker-semantic-search' ,CEOWorkerSemanticSearchView.as_view()),

    # GET ALL product margin data 
    path("get-profit-margin-data", GetProfitMarginData.as_view()),
    path("get-tax-data", TaxAvenueView.as_view()),
    path("get-ceo-worker-data", CeoWorkerView.as_view()),

    # URL OF TRACK VISITOR
    path("count-value/", GetProductVisitorCount.as_view()),
    path("admin-login", AdminAuthenticationView.as_view()),
    path("auth-check", TokenProtectedView.as_view()),
    
    # RDS API"S 
    path("sync-profit-margin-data", SyncProfitMarginDataView.as_view()),
    path("sync-tax-data", SyncTaxDataView.as_view()),
    path("sync-ceo-worker-data", SyncCEOWorkerDataView.as_view())

    ]

