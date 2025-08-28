# myapp/management/commands/import_data.py
import csv
from django.core.management.base import BaseCommand
from SearchEngineApp.models import * # Replace with your model

class Command(BaseCommand):
        help = 'Imports data from a CSV file into MyModel'

# 📌 parser is a tool that helps Django read values from the command line.

        def add_arguments(self, parser):
            parser.add_argument('csv_file', type=str, help='The path to the CSV file')

        def handle(self, *args, **options):
            csv_file_path = options['csv_file']
            with open(csv_file_path, 'r') as file:              
                reader = csv.reader(file)
                next(reader)  # Skip header row if present

                for row in reader:              
                    # Assuming CSV columns match model fields in order
                    # Adjust as per your CSV and model structure
                    ProductModel.objects.create(
                        brand=row[0],
                        product_name=row[1],
                        Type=row[2],
                        year=row[3],
                        product_url=row[4],
                        profit_price=row[5],
                        profit_made=row[6],
                        profit_margin=row[7],

                        
                        # ... and so on for other fields
                    )
            self.stdout.write(self.style.SUCCESS('Data imported successfully!'))


