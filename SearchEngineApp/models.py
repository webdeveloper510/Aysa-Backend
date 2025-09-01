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


class TaxModel(models.Model):
    company_name = models.CharField(max_length=500 , blank=True , null=True)
    year = models.CharField(max_length=500 , blank=True , null=True)
    tax_paid = models.CharField(max_length=500 , blank=True , null=True)
    tax_avoid = models.CharField(max_length = 255 ,blank=True, null=True)


class CEOWokrerModel(models.Model):
    company_name = models.CharField(max_length=500 , blank=True , null=True)
    year = models.CharField(max_length=500 , blank=True , null=True)
    ceo_name = models.CharField(max_length=500 , blank=True , null=True)
    ceo_total_compensation = models.CharField(max_length = 500 ,blank=True, null=True)
    worker_salary = models.CharField(max_length = 500 ,blank=True, null=True)



class ProductSearchTrack(models.Model):
    TAB_CHOICES = [
        ('profit', 'Profit'),
        ('tax', 'Tax'),
        ('ceo_worker', 'CEO Worker'),
    ]

    #visit_date = models.DateField()
    product_name = models.CharField(max_length=255)
    tab_type = models.CharField(max_length=20, choices=TAB_CHOICES)
    search_count = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)  # optional
    updated_at = models.DateTimeField(auto_now=True)      # optional

    class Meta:
        unique_together = ('product_name', 'tab_type')  
        # ensures each product+tab combination is unique

    def __str__(self):
        return f"{self.product_name} ({self.tab_type}) - {self.search_count}"



class AdminAuthenticationModel(models.Model):
    password = models.CharField(max_length = 1000)
    created_at = models.DateTimeField(auto_now_add = True)
    updated_at = models.DateTimeField(auto_now= True)
