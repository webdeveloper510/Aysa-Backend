from django.db import models

# Create your models here.

class ProductModel(models.Model):
    brand = models.CharField(max_length=500 , blank=True , null=True)
    product_name  = models.CharField(max_length=500 , blank=True , null=True)
    Type = models.CharField(max_length=500 , blank=True , null=True)
    year = models.CharField(max_length=500 , blank=True , null=True)
    product_url = models.CharField(max_length=500 , blank=True , null=True)
    profit_price = models.CharField(max_length = 255 ,blank=True, null=True)
    profit_made = models.CharField(max_length=255 , blank=True, null=True)
    profit_margin = models.CharField(max_length = 255,blank=True, null=True)  