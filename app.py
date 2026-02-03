import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import pubchempy as pcp
import py3Dmol
import stmol
import requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# 페이지 설정
st.set_page_config(
    page_title="Nano-Chem Studio",
    page_icon="🔬",
    layout="wide"
)

# 사이드바 메뉴
st.sidebar.title("🔬 Nano-Chem Studio")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Select Menu",
    ["SEM Analysis", "3D Chemical Lab"]
)

# 메인 화면
if menu == "SEM Analysis":
    st.title("SEM Analysis")
    st.markdown("---")
    
    # 사이드바에 파라미터 슬라이더 추가
    st.sidebar.markdown("### ⚙️ Analysis Parameters")
    threshold_value = st.sidebar.slider(
        "Threshold",
        min_value=0,
        max_value=255,
        value=127,
        help="Adjust binarization intensity. Higher values recognize only darker areas as particles."
    )
    min_area = st.sidebar.slider(
        "Min Area",
        min_value=0,
        max_value=1000,
        value=100,
        step=10,
        help="Contours smaller than this value are considered noise and excluded."
    )
    
    # 파일 업로더
    uploaded_file = st.file_uploader(
        "Upload SEM Image (JPG, PNG)",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        # 이미지 읽기
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        # OpenCV 형식으로 변환 (RGB -> BGR)
        if len(image_array.shape) == 3:
            image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            image_cv = image_array
        
        # 두 컬럼 레이아웃
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Analysis Result")
            
            # 이미지 처리
            # 1. 흑백 변환
            if len(image_cv.shape) == 3:
                gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_cv.copy()
            
            # 2. Gaussian Blur 적용 (노이즈 제거)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 3. 이진화 (사용자 지정 임계값 사용)
            _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)
            
            # 4. 윤곽선 찾기
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 5. 최소 크기 필터링
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
            
            # 6. 결과 이미지에 윤곽선 그리기 (더 두껍게)
            result_image = image_cv.copy()
            cv2.drawContours(result_image, filtered_contours, -1, (0, 255, 0), 3)
            
            # BGR -> RGB로 변환하여 표시
            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            st.image(result_image_rgb, use_container_width=True)
            
            # 입자 개수 및 평균 크기 계산
            particle_count = len(filtered_contours)
            
            if particle_count > 0:
                # 각 입자의 면적 계산 (데이터 수집)
                particle_areas = [cv2.contourArea(cnt) for cnt in filtered_contours]
                avg_area = np.mean(particle_areas)
                
                # 통계 계산
                mean_area = np.mean(particle_areas)
                std_area = np.std(particle_areas)
                max_area = np.max(particle_areas)
                min_area = np.min(particle_areas)
                
                st.markdown("---")
                st.markdown(f"### Total Particles Detected: **{particle_count}**")
                st.markdown(f"### Average Particle Size: **{avg_area:.1f} pixels**")
                
                # 통계 요약 표시
                st.markdown("---")
                st.subheader("Statistical Summary")
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                
                with col_stat1:
                    st.metric("Mean", f"{mean_area:.2f}")
                with col_stat2:
                    st.metric("Std Dev", f"{std_area:.2f}")
                with col_stat3:
                    st.metric("Max", f"{max_area:.2f}")
                with col_stat4:
                    st.metric("Min", f"{min_area:.2f}")
                
                # 인터랙티브 히스토그램 그리기 (Plotly)
                st.markdown("---")
                st.subheader("Particle Size Distribution")
                
                # Plotly 히스토그램 생성
                fig = go.Figure()
                
                fig.add_trace(go.Histogram(
                    x=particle_areas,
                    nbinsx=30,
                    marker_color='darkblue',
                    marker_line_color='navy',
                    marker_line_width=1,
                    opacity=0.8,
                    hovertemplate='<b>Area:</b> %{x:.2f} pixels<br><b>Count:</b> %{y}<extra></extra>'
                ))
                
                fig.update_layout(
                    title='Particle Size Distribution',
                    xaxis_title='Area (pixels)',
                    yaxis_title='Count',
                    template='plotly_white',
                    height=500,
                    hovermode='closest'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # CSV 다운로드 버튼
                st.markdown("---")
                st.subheader("Download Analysis Results")
                
                # 데이터프레임 생성
                df = pd.DataFrame({
                    'Particle ID': range(1, particle_count + 1),
                    'Area': particle_areas
                })
                
                # CSV로 변환
                csv = df.to_csv(index=False, encoding='utf-8-sig')
                
                st.download_button(
                    label="📥 Download Analysis Results (CSV)",
                    data=csv,
                    file_name=f"sem_particle_analysis_{uploaded_file.name.split('.')[0]}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            else:
                st.markdown("---")
                st.warning("⚠️ No particles detected matching the criteria. Please adjust Threshold or Min Area values.")
    
    else:
        st.info("👆 Please upload an SEM image file above.")

elif menu == "3D Chemical Lab":
    st.title("3D Chemical Lab")
    st.markdown("---")
    
    # CPK 색상표 정의
    CPK_COLORS = {
        'H': {'color': '#FFFFFF', 'name': 'Hydrogen', 'korean': '수소', 'border': '#CCCCCC'},
        'C': {'color': '#909090', 'name': 'Carbon', 'korean': '탄소', 'border': None},
        'N': {'color': '#3050F8', 'name': 'Nitrogen', 'korean': '질소', 'border': None},
        'O': {'color': '#FF0D0D', 'name': 'Oxygen', 'korean': '산소', 'border': None},
        'F': {'color': '#90E050', 'name': 'Fluorine', 'korean': '플루오린', 'border': None},
        'Cl': {'color': '#1FF01F', 'name': 'Chlorine', 'korean': '염소', 'border': None},
        'Br': {'color': '#A62929', 'name': 'Bromine', 'korean': '브로민', 'border': None},
        'I': {'color': '#940094', 'name': 'Iodine', 'korean': '아이오딘', 'border': None},
        'S': {'color': '#FFFF30', 'name': 'Sulfur', 'korean': '황', 'border': None},
        'P': {'color': '#FF8000', 'name': 'Phosphorus', 'korean': '인', 'border': None}
    }
    
    # 분자식에서 원소 추출 함수
    def extract_elements_from_formula(formula):
        """분자식에서 원소 기호를 추출합니다."""
        import re
        elements = set()
        # 원소 기호 패턴 (대문자로 시작하고 소문자가 올 수 있음)
        pattern = r'([A-Z][a-z]?)(\d*)'
        matches = re.findall(pattern, formula)
        for element, count in matches:
            elements.add(element)
        return sorted(elements)
    
    # SDF 파일에서 원소 추출 함수
    def extract_elements_from_sdf(sdf_content):
        """SDF 파일에서 원소 기호를 추출합니다."""
        elements = set()
        lines = sdf_content.split('\n')
        # SDF 파일의 원자 정보는 보통 4번째 라인 이후에 시작
        if len(lines) >= 4:
            try:
                # 4번째 라인에서 원자 수와 결합 수 추출
                header_parts = lines[3].split()
                if header_parts:
                    num_atoms = int(header_parts[0])
                    # 원자 정보 라인들 파싱 (일반적으로 x, y, z 좌표 다음에 원소 기호)
                    for i in range(4, min(4 + num_atoms, len(lines))):
                        parts = lines[i].strip().split()
                        if len(parts) >= 4:
                            # SDF 형식: x y z element 또는 다른 형식
                            # 원소 기호는 보통 마지막 부분에 있거나 특정 위치에 있음
                            # 일반적으로 4번째 컬럼이 원소 기호일 가능성이 높음
                            for part in parts:
                                # 원소 기호는 대문자로 시작하고 알파벳만 포함
                                if part and part[0].isupper() and part.isalpha() and len(part) <= 2:
                                    elements.add(part)
                                    break
            except (ValueError, IndexError):
                pass
        return sorted(elements)
    
    # 범례 생성 함수
    def create_legend_html(elements, cpk_colors):
        """원소 리스트로부터 범례 HTML을 생성합니다."""
        legend_items = []
        for elem in elements:
            if elem in cpk_colors:
                color_info = cpk_colors[elem]
                border_style = f"border: 2px solid {color_info['border']};" if color_info['border'] else ""
                legend_items.append(
                    f'<span style="display: inline-flex; align-items: center; margin-right: 20px; margin-bottom: 10px;">'
                    f'<span style="display: inline-block; width: 20px; height: 20px; background-color: {color_info["color"]}; '
                    f'{border_style} border-radius: 50%; margin-right: 8px;"></span>'
                    f'<strong>{elem}</strong> - {color_info["name"]} ({color_info["korean"]})'
                    f'</span>'
                )
            else:
                # 기타 원소
                legend_items.append(
                    f'<span style="display: inline-flex; align-items: center; margin-right: 20px; margin-bottom: 10px;">'
                    f'<span style="display: inline-block; width: 20px; height: 20px; background-color: #FF69B4; '
                    f'border-radius: 50%; margin-right: 8px;"></span>'
                    f'<strong>{elem}</strong> - Other'
                    f'</span>'
                )
        return '<div style="display: flex; flex-wrap: wrap; align-items: center;">' + ''.join(legend_items) + '</div>'
    
    # 검색창
    col_search1, col_search2 = st.columns([3, 1])
    
    with col_search1:
        compound_name = st.text_input(
            "Enter Chemical Compound Name (e.g., Aspirin, Caffeine)",
            placeholder="Enter chemical compound name"
        )
    
    with col_search2:
        st.markdown("<br>", unsafe_allow_html=True)  # 버튼을 텍스트 입력과 같은 높이로 맞추기
        search_button = st.button("🔍 Search", use_container_width=True)
    
    # 검색 실행
    if search_button or (compound_name and compound_name.strip()):
        if compound_name and compound_name.strip():
            try:
                with st.spinner("Fetching molecular information..."):
                    # PubChem에서 분자 검색
                    compounds = pcp.get_compounds(compound_name.strip(), 'name')
                    
                    if compounds:
                        compound = compounds[0]
                        cid = compound.cid
                        
                        # 3D 구조 가져오기 (PubChem REST API 사용)
                        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/record/SDF/?record_type=3d"
                        response = requests.get(url)
                        
                        if response.status_code == 200:
                            mol_block = response.text
                            
                            # 분자에서 원소 추출 (SDF 파일과 분자식 모두 시도)
                            elements_in_molecule = set()
                            
                            # SDF 파일에서 추출 시도
                            elements_from_sdf = extract_elements_from_sdf(mol_block)
                            elements_in_molecule.update(elements_from_sdf)
                            
                            # 분자식에서 추출 시도
                            if hasattr(compound, 'molecular_formula') and compound.molecular_formula:
                                elements_from_formula = extract_elements_from_formula(compound.molecular_formula)
                                elements_in_molecule.update(elements_from_formula)
                            
                            # 정렬된 원소 리스트
                            elements_list = sorted(elements_in_molecule) if elements_in_molecule else []
                            
                            # 3D 시각화
                            st.subheader(f"3D Structure: {compound_name.strip()}")
                            
                            # 라벨 표시 체크박스
                            show_labels = st.checkbox("원소 기호 표시 (Show Labels)", value=False)
                            
                            # py3Dmol 뷰어 생성 (정확한 순서)
                            view = py3Dmol.view(width=800, height=600)
                            view.addModel(mol_block, 'sdf')
                            view.setStyle({'stick': {'colorscheme': 'default'}})
                            view.setBackgroundColor('0x1e1e1e')  # 어두운 배경
                            
                            # 라벨 표시 기능 (체크박스 상태에 따라)
                            # 주의: addPropertyLabels는 zoomTo()와 showmol()보다 앞에 와야 함
                            if show_labels:
                                # 원소 기호를 표시 (sel: {}, prop: 'elem', style: {...})
                                view.addPropertyLabels({}, 'elem', {
                                    'fontColor': 'white',
                                    'fontSize': 16,
                                    'showBackground': True,
                                    'backgroundColor': 'black',
                                    'backgroundOpacity': 0.5,
                                    'alignment': 'center'
                                })
                            else:
                                # 라벨 제거
                                view.removeAllLabels()
                            
                            view.zoomTo()
                            view.spin(False)  # 자동 회전 비활성화 (마우스로 회전 가능)
                            
                            # Streamlit에 표시
                            stmol.showmol(view, height=600, width=800)
                            
                            # 동적 스마트 범례 추가
                            if elements_list:
                                st.markdown("---")
                                st.markdown("### 🎨 원소 색상 범례 (Element Color Legend)")
                                legend_html = create_legend_html(elements_list, CPK_COLORS)
                                st.markdown(legend_html, unsafe_allow_html=True)
                        else:
                            st.warning("Unable to fetch 3D structure. Displaying 2D structure instead.")
                            # 2D 구조로 대체 시도
                            try:
                                mol_block_2d = pcp.get_compounds(cid, 'cid')[0].record.get('3d_structure')
                                if mol_block_2d:
                                    # 분자에서 원소 추출
                                    elements_in_molecule = set()
                                    if hasattr(compound, 'molecular_formula') and compound.molecular_formula:
                                        elements_from_formula = extract_elements_from_formula(compound.molecular_formula)
                                        elements_in_molecule.update(elements_from_formula)
                                    
                                    elements_list = sorted(elements_in_molecule) if elements_in_molecule else []
                                    
                                    # 라벨 표시 체크박스
                                    show_labels = st.checkbox("원소 기호 표시 (Show Labels)", value=False, key="labels_2d")
                                    
                                    # py3Dmol 뷰어 생성 (정확한 순서)
                                    view = py3Dmol.view(width=800, height=600)
                                    view.addModel(mol_block_2d, 'mol')
                                    view.setStyle({'stick': {'colorscheme': 'default'}})
                                    view.setBackgroundColor('0x1e1e1e')
                                    
                                    # 라벨 표시 기능
                                    # 주의: addPropertyLabels는 zoomTo()와 showmol()보다 앞에 와야 함
                                    if show_labels:
                                        # 원소 기호를 표시 (sel: {}, prop: 'elem', style: {...})
                                        view.addPropertyLabels({}, 'elem', {
                                            'fontColor': 'white',
                                            'fontSize': 16,
                                            'showBackground': True,
                                            'backgroundColor': 'black',
                                            'backgroundOpacity': 0.5,
                                            'alignment': 'center'
                                        })
                                    else:
                                        view.removeAllLabels()
                                    
                                    view.zoomTo()
                                    view.spin(False)
                                    stmol.showmol(view, height=600, width=800)
                                    
                                    # 동적 스마트 범례 추가
                                    if elements_list:
                                        st.markdown("---")
                                        st.markdown("### 🎨 원소 색상 범례 (Element Color Legend)")
                                        legend_html = create_legend_html(elements_list, CPK_COLORS)
                                        st.markdown(legend_html, unsafe_allow_html=True)
                                else:
                                    st.error("Unable to fetch molecular structure.")
                            except Exception as e:
                                st.error(f"Unable to fetch molecular structure: {str(e)}")
                        
                        # 분자 정보 표시
                        st.markdown("---")
                        st.subheader("Molecular Information")
                        
                        col_info1, col_info2 = st.columns(2)
                        
                        with col_info1:
                            if hasattr(compound, 'molecular_weight') and compound.molecular_weight:
                                st.metric("Molecular Weight", f"{compound.molecular_weight:.2f} g/mol")
                            else:
                                st.metric("Molecular Weight", "N/A")
                        
                        with col_info2:
                            if hasattr(compound, 'molecular_formula') and compound.molecular_formula:
                                st.metric("Molecular Formula", compound.molecular_formula)
                            else:
                                st.metric("Molecular Formula", "N/A")
                        
                        # 추가 정보 (선택사항)
                        if hasattr(compound, 'iupac_name') and compound.iupac_name:
                            st.markdown(f"**IUPAC Name:** {compound.iupac_name}")
                        
                    else:
                        st.error(f"Compound '{compound_name}' not found. Please try a different name.")
                        
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("💡 Tip: Try using common names (e.g., Aspirin, Caffeine) or IUPAC names.")
        else:
            st.info("👆 Please enter a chemical compound name above and click the search button.")
    else:
        st.info("👆 Please enter a chemical compound name above and click the search button.")
        st.markdown("---")
        st.markdown("### 💡 Examples")
        st.markdown("- **Aspirin**")
        st.markdown("- **Caffeine**")
        st.markdown("- **Glucose**")
        st.markdown("- **Water**")
