import os
import re
import json
import pdfplumber
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import concurrent.futures
import openai

class GeotechnicalDataExtractor:
    """Extract structured geotechnical data from soil investigation reports"""
    
    def __init__(self, data_dir: str = "Data", output_file: str = "geotechnical_dataset.json", use_llm: bool = True):
        """
        Initialize the extractor with directory, output file, and LLM settings
        
        Args:
            data_dir: Directory containing PDF reports
            output_file: Path for JSON output file
            use_llm: Whether to use LLM assistance for complex extraction
        """
        self.data_dir = Path(data_dir)
        self.output_file = output_file
        self.reports = []
        self.use_llm = use_llm
        self.llm_client = None
        
        # Patterns for project ID extraction
        self.project_id_patterns = [
            r'(?:Ref\.|Reference)[\s\.:]*(ST\/)?(\d{4}[\s\-]*\d{2})',
            r'Plan\s*No[\.:]?\s*(\d{4})[\s\-]*',
            r'(?:Project|File)[:\s]+.*?(\d{4}[\s\-]*\d{2})'
        ]
        
        # Initialize LLM if enabled
        if self.use_llm:
            self.setup_llm_client()
    
    def setup_llm_client(self):
        """Initialize LLM client with API key from environment variable"""
        try:
        # Check if we have the newer OpenAI library
            try:
                from openai import OpenAI
                self.client = OpenAI()  # Automatically reads from environment
                # Test if API key works
                self.client.models.list(limit=1)
                self.use_modern_api = True
                print("✅ OpenAI API connection successful (modern API)")
            except ImportError:
                # Fall back to legacy API
                import openai
                openai.api_key = os.environ.get("OPENAI_API_KEY")
                self.use_modern_api = False
                # Test if API key works
                openai.Model.list(limit=1)
                print("✅ OpenAI API connection successful (legacy API)")
        except Exception as e:
            print(f"⚠️ OpenAI API setup failed: {e}")
            self.use_llm = False

    def llm_extract_complex_data(self, text: str, instruction: str) -> Optional[Dict]:
        """Use LLM to extract complex data from text"""
        if not self.use_llm:
            return None
            
        try:
            # Truncate text if too long
            max_tokens = 12000
            if len(text) > max_tokens:
                text = text[:max_tokens] + "...[text truncated]"
            
            if hasattr(self, 'use_modern_api') and self.use_modern_api:
                # Use modern API format
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a geotechnical data extraction assistant. Extract the requested information in JSON format."},
                        {"role": "user", "content": f"{instruction}\n\nText: {text}"}
                    ],
                    response_format={"type": "json_object"}
                )
                return json.loads(response.choices[0].message.content)
            else:
                # Use legacy API format
                response = openai.ChatCompletion.create(
                    model="gpt-4",  # Legacy API may not have gpt-4o
                    messages=[
                        {"role": "system", "content": "You are a geotechnical data extraction assistant. Extract the requested information in JSON format. Donot miss any point and search properly and thoroughly. The data is critical and it should not miss out."},
                        {"role": "user", "content": f"{instruction}\n\nText: {text}"}
                    ],
                )
                return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"⚠️ LLM extraction failed: {e}")
            return None
        
    def llm_extract_complex_data(self, text: str, instruction: str) -> Optional[Dict]:
        """
        Use LLM to extract complex data from text
        
        Args:
            text: Text to extract data from
            instruction: Instructions for what data to extract
            
        Returns:
            Extracted data in dictionary format or None if extraction failed
        """
        if not self.use_llm:
            return None
            
        try:
            # Truncate text if too long
            max_tokens = 12000
            if len(text) > max_tokens:
                text = text[:max_tokens] + "...[text truncated]"
                
            response = openai.ChatCompletion.create(
                model="gpt-4.1",  # or another suitable model
                messages=[
                    {"role": "system", "content": "You are a geotechnical data extraction assistant. Extract the requested information in JSON format. Donot miss any point and search properly and thoroughly. The data is critical and it should not miss out."},
                    {"role": "user", "content": f"{instruction}\n\nText: {text}"}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"⚠️ LLM extraction failed: {e}")
            return None
        
    def process_all_reports(self, max_workers: int = 4) -> None:
        """
        Process all PDF files in the data directory using parallel processing
        
        Args:
            max_workers: Number of parallel processes to use
        """
        pdf_files = list(self.data_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"❌ No PDF files found in {self.data_dir}")
            return
        
        print(f"🔍 Found {len(pdf_files)} PDF files to process")
        
        # Process files in parallel for speed
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_report, pdf_file) for pdf_file in pdf_files]
            
            # Collect results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    result = future.result()
                    if result:
                        self.reports.append(result)
                    print(f"✅ Processed {i+1}/{len(pdf_files)} files")
                except Exception as e:
                    print(f"❌ Error processing file: {e}")
        
        # Save the collected data
        self.save_data()
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from PDF with section markers
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Extracted text with section markers
        """
        full_text = ""
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Add page marker
                    full_text += f"\n\n==== PAGE {page_num+1} ====\n\n"
                    
                    # Extract text
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
                    
                    # Enhanced table extraction
                    tables = page.extract_tables()
                    if tables:
                        full_text += f"\n==== TABLES ON PAGE {page_num+1} ====\n"
                        for table_idx, table in enumerate(tables):
                            full_text += f"\n-- TABLE {table_idx+1} --\n"
                            for row in table:
                                if row:
                                    clean_row = [str(cell).strip() if cell else "" for cell in row]
                                    full_text += " | ".join(clean_row) + "\n"
                    
                    # Mark important sections
                    if "SUMMARY & RECOMMENDATIONS" in text:
                        full_text += "\n==== RECOMMENDATIONS SECTION DETECTED ====\n"
                    if "ABSTRACT DATA ON LABORATORY TEST RESULTS" in text:
                        full_text += "\n==== LAB RESULTS SECTION DETECTED ====\n"
            
            return full_text
            
        except Exception as e:
            print(f"❌ Error extracting text from PDF {pdf_path.name}: {e}")
            return ""

    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract key sections from report text
        
        Args:
            text: Full text from PDF
            
        Returns:
            Dictionary of extracted sections
        """
        sections = {}
        
        # Extract main sections using regex patterns
        intro_match = re.search(r'1\.0\s+INTRODUCTION(.*?)(?=2\.0|\Z)', text, re.DOTALL)
        if intro_match:
            sections["introduction"] = intro_match.group(1).strip()
        
        field_work_match = re.search(r'2\.0\s+DESCRIPTION\s+OF\s+SOIL\s+INVESTIGATION(.*?)(?=3\.0|\Z)', text, re.DOTALL)
        if field_work_match:
            sections["field_work"] = field_work_match.group(1).strip()
            
        lab_test_match = re.search(r'3\.0\s+LABORATORY\s+TEST(.*?)(?=4\.0|\Z)', text, re.DOTALL)
        if lab_test_match:
            sections["lab_test"] = lab_test_match.group(1).strip()
        
        nature_match = re.search(r'4\.0\s+NATURE\s+OF\s+SUB-SURFACE(.*?)(?=5\.0|\Z)', text, re.DOTALL)
        if nature_match:
            sections["subsurface"] = nature_match.group(1).strip()
            
        summary_match = re.search(r'5\.0\s+SUMMARY\s+&\s+RECOMMENDATIONS(.*?)(?=APPENDICES|\Z)', text, re.DOTALL)
        if summary_match:
            sections["summary"] = summary_match.group(1).strip()
            
        # Extract lab results table section
        lab_results_match = re.search(r'ABSTRACT DATA ON LABORATORY TEST RESULTS(.*?)(?=GRAIN SIZE|\Z)', text, re.DOTALL)
        if lab_results_match:
            sections["lab_results_table"] = lab_results_match.group(1).strip()
            
        # Look for specific tables
        bore_logs_match = re.search(r'(BH\.?\s*No\.?\s*1[\s\S]*?)(?=SWIS-TECH|\Z)', text, re.DOTALL)
        if bore_logs_match:
            sections["bore_logs"] = bore_logs_match.group(1).strip()
        
        # Use LLM to extract sections if regex fails
        if self.use_llm and (not sections or len(sections) < 3):
            print("⚙️ Using LLM to identify document sections...")
            llm_sections = self.llm_extract_complex_data(
                text[:5000],  # First portion of document for section identification
                "Identify the main sections in this geotechnical report. Return a JSON with keys 'introduction', 'field_work', 'lab_test', 'subsurface', 'summary', and 'bore_logs'. For each key, provide the section text."
            )
            if llm_sections:
                # Merge with regex sections, prioritizing regex results
                for key, value in llm_sections.items():
                    if key not in sections and value:
                        sections[key] = value
                        print(f"  ✓ LLM identified section: {key}")
            
        return sections
    
    def extract_project_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract project metadata (client, location, etc.)
        
        Args:
            text: Full PDF text
            
        Returns:
            Dictionary of project metadata
        """
        metadata = {}
        
        # Extract project ID using multiple patterns
        for pattern in self.project_id_patterns:
            project_id_match = re.search(pattern, text, re.IGNORECASE)
            if project_id_match:
                # Use the last group, which should contain the ID
                raw_id = project_id_match.group(project_id_match.lastindex)
                # Clean up the ID
                project_id = re.sub(r'\s+', '-', raw_id.strip())
                metadata["project_id"] = project_id
                break
                
        # Extract client name
        client_match = re.search(r'(?:PROPOSED|FOR THE PROPOSED).*?OF\s+(M[RS]+\.?\s+[\w\s\.]+?)(?:\n|AT)', text, re.IGNORECASE)
        if client_match:
            metadata["client"] = client_match.group(1).strip()
            
        # Extract project type
        project_type_match = re.search(r'(PROPOSED\s+[\w\s]+)(?:\n|AT)', text)
        if project_type_match:
            metadata["project_type"] = project_type_match.group(1).strip()
            
        # Extract location
        location_match = re.search(r'AT\s+(.*?)(?:\n\n|\n[A-Z]{3,}|\Z)', text)
        if location_match:
            metadata["location"] = location_match.group(1).strip()
            
        # Extract report date
        date_match = re.search(r'Date\.?\s*(\d{2}-\d{2}-\d{4})', text)
        if date_match:
            metadata["report_date"] = date_match.group(1).strip()
        
        # Use LLM for metadata extraction if regex fails
        if self.use_llm and (len(metadata) < 3):
            print("⚙️ Using LLM to extract project metadata...")
            llm_metadata = self.llm_extract_complex_data(
                text[:3000],  # First portion of document for metadata
                "Extract project metadata from this geotechnical report. Return a JSON with keys 'project_id', 'client', 'project_type', 'location', and 'report_date'."
            )
            if llm_metadata:
                # Merge with regex metadata, prioritizing regex results
                for key, value in llm_metadata.items():
                    if key not in metadata and value:
                        metadata[key] = value
                        print(f"  ✓ LLM identified metadata: {key}")
            
        return metadata
    
    def extract_bearing_capacity_regex(self, summary_section: str) -> Dict[str, Any]:
            """Extract bearing capacity values with enhanced table recognition"""
            result = {
                "value": None,
                "unit": "Tonne/ft²",
                "depth": None,
                "alternative_values": []
            }
            
            # Enhanced table pattern for 7155-style format
            table_pattern = r'(?:Depth of Footing|Foundation)[^\n]*\n[^\n]*\n(?:[^\n]*\n)*?.*?(\d+[\'\"]-\d+[\'\"]).+?(\d+\.\d+)\s*Tonne\/ft[²2]'
            # Simpler alternative pattern that looks for key phrases
            alt_pattern = r'(?:bearing capacity|allowable bearing)[^.]*?(\d+\.\d+)\s*Tonne\/ft[²2]'
            # Specific pattern for the 7155 report format with depth patterns
            specific_pattern = r'(\d+[\'\"]-\d+[\'\"])\s*or below\s+[-\d\.]+\s*ft\s*or below\s+(\d+\.\d+)\s*Tonne\/ft[²2]'
            
            # Try specific pattern first (highest precedence)
            specific_matches = re.findall(specific_pattern, summary_section, re.DOTALL)
            if specific_matches:
                # Use first match as primary
                result["depth"] = specific_matches[0][0]
                result["value"] = float(specific_matches[0][1])
                
                # Additional values as alternatives
                for depth, value in specific_matches[1:]:
                    result["alternative_values"].append({
                        "depth": depth,
                        "value": float(value),
                        "unit": "Tonne/ft²"
                    })
                return result
                
            # Try standard table pattern next
            table_matches = re.findall(table_pattern, summary_section, re.DOTALL)
            if table_matches:
                # Same logic as before
                result["depth"] = table_matches[0][0]
                result["value"] = float(table_matches[0][1])
                
                for depth, value in table_matches[1:]:
                    result["alternative_values"].append({
                        "depth": depth,
                        "value": float(value),
                        "unit": "Tonne/ft²"
                    })
                return result
                
            # Fall back to simplest pattern
            alt_match = re.search(alt_pattern, summary_section)
            if alt_match:
                result["value"] = float(alt_match.group(1))
                
            # Try to find depth separately
            depth_match = re.search(r'(?:at|depth of)[\s\.]+(\d+[\'\"]-\d+[\'\"])', summary_section, re.IGNORECASE)
            if depth_match:
                result["depth"] = depth_match.group(1)
            
            return result
    
    def extract_bearing_capacity(self, summary_section: str) -> Dict[str, Any]:
        """
        Extract bearing capacity values and depths with LLM fallback
        
        Args:
            summary_section: Text from summary and recommendations section
            
        Returns:
            Dictionary with bearing capacity data
        """
        # First try regex extraction
        result = self.extract_bearing_capacity_regex(summary_section)
        
        # If regex extraction failed or is incomplete, try LLM
        if self.use_llm and (result["value"] is None or result["depth"] is None):
            print("⚙️ Using LLM to extract bearing capacity...")
            llm_result = self.llm_extract_complex_data(
                summary_section,
                "Extract the bearing capacity information from this text. Return a JSON with: "
                "value (number), unit (string), depth (string), and alternative_values (array of objects with value, unit, depth)"
            )
            
            if llm_result and "value" in llm_result and llm_result["value"] is not None:
                # Use LLM result, but keep any valid fields from regex
                for key, value in llm_result.items():
                    if key in result and (result[key] is None or (key == "alternative_values" and not result[key])):
                        result[key] = value
                print("  ✓ LLM extracted bearing capacity")
                
        return result
    
    def extract_foundation_type(self, summary_section: str) -> Optional[str]:
        """Extract recommended foundation type with improved pattern matching"""
        # Enhanced patterns
        if re.search(r'adopt.*?(?:RCC\s+Strip\s*/\s*Isolated|Strip\s*/\s*Isolated)', summary_section, re.IGNORECASE):
            return "Strip/Isolated"
        elif re.search(r'adopt.*?RCC\s+Strip', summary_section, re.IGNORECASE):
            return "Strip"
        elif re.search(r'adopt.*?Raft', summary_section, re.IGNORECASE):
            return "Raft"
        elif re.search(r'adopt.*?Pile', summary_section, re.IGNORECASE):
            return "Pile"
        elif re.search(r'adopt.*?Isolated', summary_section, re.IGNORECASE):
            return "Isolated"
        
        # Use LLM with improved prompt
        if self.use_llm:
            print("⚙️ Using LLM to identify foundation type...")
            llm_result = self.llm_extract_complex_data(
                summary_section,
                "Extract the recommended foundation type from this text. Look for phrases like 'it is recommended to adopt', 'recommended foundation type', etc. Return a JSON with a single key 'foundation_type' with values like 'Strip', 'Raft', 'Pile', 'Isolated', 'Strip/Isolated', or other combinations. If not specified, return null."
            )
            
            if llm_result and "foundation_type" in llm_result:
                print(f"  ✓ LLM identified foundation type: {llm_result['foundation_type']}")
                return llm_result["foundation_type"]
                    
        return None
    
    def extract_grain_size_analysis(self, lab_results: str) -> List[Dict[str, Any]]:
        """Extract grain size analysis with improved pattern matching"""
        results = []
        
        # Enhanced pattern for "ABSTRACT DATA ON LABORATORY TEST RESULTS" table
        abstract_pattern = r'(BH\s*-\s*\d+)\s+(\d+[\'\"]-\d+[\'\"])[^%]*?GRAVEL\s*%\s*(?:NON-SLAKING|([a\d]+))'
        
        # First try to extract from abstract data table
        rows = re.findall(abstract_pattern, lab_results, re.DOTALL)
        for bh, depth, gravel_val in rows:
            bh_id = bh.strip()
            # Handle cases where gravel value is marked with 'a' (undissolved rock fragments)
            gravel = None
            if gravel_val:
                # Remove 'a' prefix if present
                gravel = int(re.sub(r'^a', '', gravel_val)) if gravel_val else None
                
            results.append({
                "borehole": bh_id,
                "depth": depth.strip(),
                "gravel": gravel,
                "sand": None,  # These values often aren't in the abstract table
                "fines": None
            })
        
        # If no results from abstract table, try another extraction method
        if not results and self.use_llm:
            print("⚙️ Using LLM to extract grain size analysis...")
            llm_result = self.llm_extract_complex_data(
                lab_results,
                "Extract grain size analysis data from this text. Pay special attention to tables with 'ABSTRACT DATA ON LABORATORY TEST RESULTS' and 'GRAIN SIZE DISTRIBUTION' sections. Look for values marked with 'a' which indicate undissolved rock fragments. Return a JSON array of objects with keys: 'borehole' (string), 'depth' (string), 'gravel' (number or null), 'sand' (number or null), 'fines' (number or null)."
            )
            
            if llm_result and isinstance(llm_result, list):
                print(f"  ✓ LLM extracted grain size data: {len(llm_result)} entries")
                return llm_result
            elif llm_result and "grain_size_analysis" in llm_result:
                print(f"  ✓ LLM extracted grain size data: {len(llm_result['grain_size_analysis'])} entries")
                return llm_result["grain_size_analysis"]
                
        return results
    
    def extract_moisture_content(self, lab_results: str) -> List[Dict[str, Any]]:
        """
        Extract moisture content from lab results table
        
        Args:
            lab_results: Text containing lab results table
            
        Returns:
            List of moisture content data
        """
        results = []
        
        # Look for table rows with BH and depth information
        rows = re.findall(r'(BH\s*[-\s]\s*\d+)\s+(\d+[\'\"]-\d+[\'\"])[^%]*MOISTURE\s+CONTENT\s*%\s*(\d+\.\d+)', lab_results, re.DOTALL)
        
        for bh, depth, moisture in rows:
            bh_id = bh.replace(" ", "-")
            results.append({
                "borehole": bh_id,
                "depth": depth.strip(),
                "value": float(moisture)
            })
        
        # Use LLM if regex finds no results
        if not results and self.use_llm:
            print("⚙️ Using LLM to extract moisture content data...")
            llm_result = self.llm_extract_complex_data(
                lab_results,
                "Extract moisture content data from this text. Return a JSON array of objects with keys: 'borehole' (string), 'depth' (string), 'value' (number)."
            )
            
            if llm_result and isinstance(llm_result, list):
                print(f"  ✓ LLM extracted moisture content data: {len(llm_result)} entries")
                return llm_result
            elif llm_result and "moisture_content" in llm_result:
                print(f"  ✓ LLM extracted moisture content data: {len(llm_result['moisture_content'])} entries")
                return llm_result["moisture_content"]
            
        return results
    
    def extract_atterberg_limits(self, lab_results: str) -> Union[str, List[Dict[str, Any]]]:
        """
        Extract Atterberg limits from lab results table
        
        Args:
            lab_results: Text containing lab results table
            
        Returns:
            List of Atterberg limits data or string message if not found
        """
        results = []
        
        # Look for table rows with BH, depth, and limits information
        rows = re.findall(r'(BH\s*[-\s]\s*\d+)\s+(\d+[\'\"]-\d+[\'\"])[^%]*LIQUID\s+LIMIT\s*(\d+\.\d+)[^%]*PLASTIC\s+LIMIT\s*(\d+\.\d+)[^%]*PLASTIC\s+(?:INDEX|LIMIT)\s*(\d+\.\d+)', lab_results, re.DOTALL)
        
        if not rows:
            # Use LLM if regex finds no results
            if self.use_llm:
                print("⚙️ Using LLM to extract Atterberg limits...")
                llm_result = self.llm_extract_complex_data(
                    lab_results,
                    "Extract Atterberg limits data from this text. Return a JSON array of objects with keys: 'borehole' (string), 'depth' (string), 'll' (number), 'pl' (number), 'pi' (number). If no Atterberg limits data exists, return {\"status\": \"not measured\"}."
                )
                
                if llm_result:
                    if isinstance(llm_result, list) and len(llm_result) > 0:
                        print(f"  ✓ LLM extracted Atterberg limits: {len(llm_result)} entries")
                        return llm_result
                    elif "atterberg_limits" in llm_result and isinstance(llm_result["atterberg_limits"], list):
                        print(f"  ✓ LLM extracted Atterberg limits: {len(llm_result['atterberg_limits'])} entries")
                        return llm_result["atterberg_limits"]
                    elif "status" in llm_result and llm_result["status"] == "not measured":
                        return "Not measured"
                        
            return "Not measured"
            
        for bh, depth, ll, pl, pi in rows:
            bh_id = bh.replace(" ", "-")
            results.append({
                "borehole": bh_id,
                "depth": depth.strip(),
                "ll": float(ll),
                "pl": float(pl),
                "pi": float(pi)
            })
            
        return results
    
    def extract_spt_values(self, bore_logs: str) -> List[Dict[str, Dict]]:
        """
        Extract SPT values from bore logs
        
        Args:
            bore_logs: Text containing bore logs
            
        Returns:
            Dictionary of borehole SPT data
        """
        boreholes = []
        
        # Split into individual borehole logs
        bh_sections = re.split(r'BH\.?\s*No\.?\s*\d+', bore_logs)
        bh_headers = re.findall(r'(BH\.?\s*No\.?\s*\d+)', bore_logs)
        
        if len(bh_sections) <= 1:
            # Use LLM if regex parsing fails
            if self.use_llm:
                print("⚙️ Using LLM to extract borehole data...")
                llm_result = self.llm_extract_complex_data(
                    bore_logs,
                    "Extract borehole data from this text. Return a JSON array where each item represents a borehole with keys: 'id' (string), 'depth' (string), 'elevation' (string), 'spt_values' (array of objects with depth and value), 'soil_profile' (array of objects with depth and description)."
                )
                
                if llm_result and isinstance(llm_result, list):
                    print(f"  ✓ LLM extracted borehole data: {len(llm_result)} boreholes")
                    return llm_result
                elif llm_result and "boreholes" in llm_result:
                    print(f"  ✓ LLM extracted borehole data: {len(llm_result['boreholes'])} boreholes")
                    return llm_result["boreholes"]
                    
            return boreholes
            
        # Skip first section (usually empty)
        for i, (header, section) in enumerate(zip(bh_headers, bh_sections[1:])):
            bh_id = f"BH-{i+1}"
            bh_data = {
                "id": bh_id,
                "depth": None,
                "elevation": None,
                "spt_values": [],
                "soil_profile": []
            }
            
            # Extract elevation
            elevation_match = re.search(r'(-?\d+(?:\.\d+)?)\s*ft', section)
            if elevation_match:
                bh_data["elevation"] = f"{elevation_match.group(1)} ft"
                
            # Extract SPT values
            spt_matches = re.findall(r'(\d+[\'\"]-\d+[\'\"])[^\n]*?(\d+\s*\+\s*\d+\s*\+\s*\d+|Refusal|\d+\s*\+\s*\d+\/\d+[\'\"])', section)
            
            for depth, value in spt_matches:
                # Clean up the value
                if "Refusal" in value:
                    spt_value = "Refusal"
                elif "+" in value and "/" not in value:
                    # Parse N-value from SPT blows (e.g., "11 + 22 + 29")
                    parts = re.findall(r'\d+', value)
                    if len(parts) >= 2:
                        spt_value = int(parts[1]) + int(parts[2])
                    else:
                        spt_value = value
                else:
                    spt_value = value
                    
                bh_data["spt_values"].append({
                    "depth": depth.strip(),
                    "value": spt_value
                })
                
            # Determine max depth from SPT values
            if bh_data["spt_values"]:
                max_depth = max(spt["depth"] for spt in bh_data["spt_values"])
                bh_data["depth"] = max_depth
                
            # Extract soil profile
            soil_matches = re.findall(r'(\d+[\'\"]-\d+[\'\"])\s*\/\s*(\d+[\'\"]-\d+[\'\"])\s*:?\s*([^:]*?)(?=\d+[\'\"]-\d+[\'\"]\/|\n\d|\Z)', section, re.DOTALL)
            
            for start_depth, end_depth, description in soil_matches:
                clean_desc = re.sub(r'\s+', ' ', description.strip())
                if clean_desc:
                    bh_data["soil_profile"].append({
                        "depth": f"{start_depth}/{end_depth}",
                        "description": clean_desc
                    })
                    
            boreholes.append(bh_data)
            
        return boreholes
    
    def extract_water_table(self, subsurface_section: str) -> str:
        """Extract groundwater table information with better specificity"""
        # Look for definitive statements about GWT
        definitive_pattern = r'Ground\s+Water\s+Table\s*\(GWT\)[^.]*?(?:is|was)\s+not\s+encountered[^.]*\.'
        definitive_match = re.search(definitive_pattern, subsurface_section, re.IGNORECASE)
        
        if definitive_match:
            return definitive_match.group(0).strip()
        
        # Alternative patterns
        alt_patterns = [
            r'No\s+groundwater\s+table\s+was\s+encountered',
            r'No\s+GWT\s+was\s+encountered',
            r'GWT\s+is\s+not\s+encountered'
        ]
        
        for pattern in alt_patterns:
            match = re.search(pattern, subsurface_section, re.IGNORECASE)
            if match:
                return match.group(0).strip()
        
        # Use LLM with improved prompt
        if self.use_llm:
            print("⚙️ Using LLM to extract ground water table info...")
            llm_result = self.llm_extract_complex_data(
                subsurface_section,
                "Extract information about the ground water table (GWT) from this text. Look for definitive statements about whether GWT was encountered or not. Return a JSON with a single key 'ground_water_table' containing the exact sentence describing the ground water conditions. If not explicitly mentioned, return null."
            )
            
            if llm_result and "ground_water_table" in llm_result:
                print(f"  ✓ LLM identified ground water table info")
                return llm_result["ground_water_table"]
                    
        return None
    
    def extract_soil_properties(self, subsurface_section: str) -> Dict[str, Any]:
        """
        Extract soil description properties
        
        Args:
            subsurface_section: Text containing subsurface description
            
        Returns:
            Dictionary of soil properties
        """
        properties = {
            "consistency": None,
            "soil_color": None,
            "moisture_condition": None,
            "uscs_classification": None
        }
        
        # Extract USCS classification
        uscs_match = re.search(r'(?:classified|classification)[^.]*?\(((?:[A-Z]{2})(?:-[A-Z]{2})?)\)', subsurface_section)
        if uscs_match:
            # Take the first classification if hyphenated
            classification = uscs_match.group(1).split('-')[0]
            properties["uscs_classification"] = classification
            
        # Extract consistency
        for term in ["soft", "firm", "stiff", "hard", "very hard", "loose", "dense", "very dense"]:
            if term in subsurface_section.lower():
                properties["consistency"] = term.replace("very ", "")
                break
                
        # Extract color
        for color in ["brown", "reddish brown", "gray", "grey", "red", "yellow", "black"]:
            if color in subsurface_section.lower():
                properties["soil_color"] = "brown" if "brown" in color else color
                break
                
        # Extract moisture condition
        for condition in ["dry", "moist", "slightly moist", "wet", "saturated", "damp"]:
            if condition in subsurface_section.lower():
                if "slightly" in condition:
                    properties["moisture_condition"] = "slightly moist"
                else:
                    properties["moisture_condition"] = condition
                break
        
        # Use LLM if regex fails to identify key properties
        if self.use_llm and (properties["consistency"] is None or 
                           properties["soil_color"] is None or 
                           properties["uscs_classification"] is None):
            print("⚙️ Using LLM to extract soil properties...")
            llm_result = self.llm_extract_complex_data(
                subsurface_section,
                "Extract soil properties from this text. Return a JSON with keys: 'consistency', 'soil_color', 'moisture_condition', and 'uscs_classification'."
            )
            
            if llm_result:
                # Update missing properties from LLM results
                for key in properties:
                    if properties[key] is None and key in llm_result and llm_result[key]:
                        properties[key] = llm_result[key]
                        print(f"  ✓ LLM identified {key}: {llm_result[key]}")
                
        return properties
    
    def calculate_average_values(self, data: Dict) -> Dict[str, float]:
        """
        Calculate average values for numerical parameters
        
        Args:
            data: Dictionary containing extracted data
            
        Returns:
            Dictionary of average values
        """
        avg_values = {}
        
        # Calculate average SPT value
        if "boreholes" in data:
            valid_spt_values = []
            for bh in data["boreholes"]:
                for spt in bh["spt_values"]:
                    if isinstance(spt["value"], (int, float)):
                        valid_spt_values.append(spt["value"])
            
            if valid_spt_values:
                avg_values["spt_n_value"] = sum(valid_spt_values) / len(valid_spt_values)
            
        # Calculate average grain size values
        if "laboratory_results" in data and "grain_size_analysis" in data["laboratory_results"]:
            gravel_values = []
            sand_values = []
            fines_values = []
            
            for item in data["laboratory_results"]["grain_size_analysis"]:
                if isinstance(item["gravel"], (int, float)):
                    gravel_values.append(item["gravel"])
                if isinstance(item["sand"], (int, float)):
                    sand_values.append(item["sand"])
                if isinstance(item["fines"], (int, float)):
                    fines_values.append(item["fines"])
                    
            if gravel_values:
                avg_values["gravel_pct"] = sum(gravel_values) / len(gravel_values)
            if sand_values:
                avg_values["sand_pct"] = sum(sand_values) / len(sand_values)
            if fines_values:
                avg_values["fines_pct"] = sum(fines_values) / len(fines_values)
                
        # Calculate average moisture content
        if "laboratory_results" in data and "moisture_content" in data["laboratory_results"]:
            moisture_values = []
            
            for item in data["laboratory_results"]["moisture_content"]:
                moisture_values.append(item["value"])
                
            if moisture_values:
                avg_values["moisture_content_pct"] = sum(moisture_values) / len(moisture_values)
                
        # Calculate average Atterberg limits
        if ("laboratory_results" in data and 
            "atterberg_limits" in data["laboratory_results"] and 
            isinstance(data["laboratory_results"]["atterberg_limits"], list)):
            
            ll_values = []
            pl_values = []
            pi_values = []
            
            for item in data["laboratory_results"]["atterberg_limits"]:
                ll_values.append(item["ll"])
                pl_values.append(item["pl"])
                pi_values.append(item["pi"])
                
            if ll_values:
                avg_values["liquid_limit_ll"] = sum(ll_values) / len(ll_values)
            if pl_values:
                avg_values["plastic_limit_pl"] = sum(pl_values) / len(pl_values)
            if pi_values:
                avg_values["plasticity_index"] = sum(pi_values) / len(pi_values)
                
        return avg_values
    
    def validate_extracted_data(self, data: Dict) -> Dict:
        """
        Use LLM to validate and fix obvious errors in data
        
        Args:
            data: Dictionary with extracted data
            
        Returns:
            Validated and potentially corrected data
        """
        if not self.use_llm:
            return data
            
        try:
            print("⚙️ Validating extracted data with LLM...")
            
            # Convert relevant part to JSON string for the LLM
            data_json = json.dumps(data["geotechnical_data"], indent=2)
            
            validation_result = self.llm_extract_complex_data(
                data_json,
                "Review this geotechnical data for inconsistencies or errors. "
                "Return a JSON with the same structure but with corrections where needed. "
                "Focus on fixing obviously wrong values, units, or classifications."
            )
            
            if validation_result:
                print("  ✓ LLM validation complete")
                data["geotechnical_data"] = validation_result
                data["validation_performed"] = True
                
            return data
        except Exception as e:
            print(f"  ⚠️ LLM validation failed: {e}")
            return data
    
    def preprocess_pdf_sections(self, text: str) -> Dict[str, str]:
        """
        Apply sophisticated pre-processing to identify document structure
        before extraction attempts
        """
        # First try with LLM - it's better at understanding document structure
        if self.use_llm:
            print("⚙️ Pre-processing document structure with LLM...")
            llm_result = self.llm_extract_complex_data(
                text[:8000],  # Use a larger chunk for better context
                """
                Analyze this geotechnical report and identify its structure. 
                Return a JSON with the following keys:
                1. 'document_type': What kind of geotechnical document this is
                2. 'table_format': How tables are formatted (e.g., 'standard', 'complex', 'minimal')
                3. 'key_pages': Map important sections to page numbers
                4. 'extraction_strategy': Recommended approach for data extraction
                5. 'expected_data': What data fields we should expect to find
                """
            )
            
            if llm_result:
                print(f"  ✓ Document structure analysis complete")
                # Store this metadata to guide later extraction
                self.doc_structure = llm_result
                
                # Adapt extraction strategies based on document structure
                if llm_result.get("table_format") == "complex":
                    # Use more sophisticated table extraction for complex formats
                    print("  ℹ️ Using enhanced table extraction for complex document format")
                    # ... code to adjust extraction methods ...
                    
        # Continue with regular extraction
        return self.extract_sections(text)
    
    def cross_validate_data(self, data: Dict) -> Dict:
        """
        Cross-validate different data points for consistency
        
        For example: Check if SPT values match soil descriptions,
        validate moisture content against soil conditions, etc.
        """
        try:
            # Get geotechnical data
            geotech = data["geotechnical_data"]
            
            # Cross-check bearing capacity with SPT values
            if geotech.get("bearing_capacity", {}).get("value") is not None:
                # Get average SPT N-value (if available)
                spt_values = []
                for bh in geotech.get("boreholes", []):
                    for spt in bh.get("spt_values", []):
                        if isinstance(spt.get("value"), (int, float)):
                            spt_values.append(spt["value"])
                
                if spt_values and geotech["bearing_capacity"]["value"] > 0:
                    avg_spt = sum(spt_values) / len(spt_values)
                    expected_bc_range = (avg_spt * 0.04, avg_spt * 0.15)  # Simplified empirical relation
                    
                    # If bearing capacity is outside expected range, flag it
                    actual_bc = geotech["bearing_capacity"]["value"]
                    if actual_bc < expected_bc_range[0] or actual_bc > expected_bc_range[1]:
                        data["validation_issues"] = data.get("validation_issues", [])
                        data["validation_issues"].append({
                            "field": "bearing_capacity",
                            "issue": f"Value {actual_bc} seems inconsistent with average SPT N-value {avg_spt}",
                            "expected_range": expected_bc_range
                        })
            
            return data
        except Exception as e:
            print(f"⚠️ Cross-validation failed: {e}")
            return data

    def process_report(self, pdf_path: Path) -> Optional[Dict]:
        """
        Process a single geotechnical report PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extracted data or None if processing failed
        """
        try:
            print(f"\n🔍 Processing {pdf_path.name}")
            
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_path)
            if not text:
                print(f"❌ Failed to extract text from {pdf_path.name}")
                return None
                
            # Extract sections
            sections = self.extract_sections(text)
            if not sections:
                print(f"❌ Failed to identify sections in {pdf_path.name}")
                
            # Extract metadata
            metadata = self.extract_project_metadata(text)
            
            # Initialize data structure
            data = {
                "project_id": metadata.get("project_id", pdf_path.stem),
                "client": metadata.get("client", "Unknown"),
                "project_type": metadata.get("project_type", "Unknown"),
                "location": metadata.get("location", "Unknown"),
                "report_date": metadata.get("report_date", "Unknown"),
                "geotechnical_data": {}
            }
            
            # Extract bearing capacity
            if "summary" in sections:
                bearing_capacity = self.extract_bearing_capacity(sections["summary"])
                data["geotechnical_data"]["bearing_capacity"] = bearing_capacity
                
                # Extract foundation type
                foundation_type = self.extract_foundation_type(sections["summary"])
                data["geotechnical_data"]["foundation_type"] = foundation_type
            
            # Extract boreholes data
            if "bore_logs" in sections:
                boreholes = self.extract_spt_values(sections["bore_logs"])
                data["geotechnical_data"]["boreholes"] = boreholes
            
            # Extract laboratory results
            lab_results = {}
            
            if "lab_results_table" in sections:
                # Extract grain size analysis
                grain_size = self.extract_grain_size_analysis(sections["lab_results_table"])
                lab_results["grain_size_analysis"] = grain_size
                
                # Extract moisture content
                moisture = self.extract_moisture_content(sections["lab_results_table"])
                lab_results["moisture_content"] = moisture
                
                # Extract Atterberg limits
                atterberg = self.extract_atterberg_limits(sections["lab_results_table"])
                lab_results["atterberg_limits"] = atterberg
            
            data["geotechnical_data"]["laboratory_results"] = lab_results
            
            # Extract water table information
            if "subsurface" in sections:
                water_table = self.extract_water_table(sections["subsurface"])
                data["geotechnical_data"]["ground_water_table"] = water_table
                
                # Extract soil properties
                soil_props = self.extract_soil_properties(sections["subsurface"])
                data["geotechnical_data"].update(soil_props)
            
            # Calculate average values
            avg_values = self.calculate_average_values(data["geotechnical_data"])
            data["geotechnical_data"].update(avg_values)
            
            # Validate data with LLM
            if self.use_llm:
                data = self.validate_extracted_data(data)
            
            print(f"✅ Successfully processed {pdf_path.name}")
            return data
            
        except Exception as e:
            print(f"❌ Error processing {pdf_path.name}: {str(e)}")
            return None
    
    def save_data(self) -> None:
        """Save extracted data to JSON file"""
        with open(self.output_file, 'w') as f:
            json.dump({"reports": self.reports}, f, indent=2)
        
        print(f"\n💾 Saved {len(self.reports)} reports to {self.output_file}")


def main():
    """Main entry point"""
    print("🌟 Geotechnical Data Extraction Tool 🌟")
    print("=" * 40)
    
    
    use_llm = os.environ.get("OPENAI_API_KEY") is not None
    if not use_llm:
        print("⚠️ OpenAI API key not found in environment. LLM features will be disabled.")
        print("   Set the OPENAI_API_KEY environment variable to enable LLM assistance.")
    
    
    extractor = GeotechnicalDataExtractor(
        data_dir="Data",
        output_file="geotechnical_dataset.json",
        use_llm=use_llm
    )
    
    
    extractor.process_all_reports(max_workers=4)
    
    print("\n🎉 Processing complete!")


if __name__ == "__main__":
    main()

