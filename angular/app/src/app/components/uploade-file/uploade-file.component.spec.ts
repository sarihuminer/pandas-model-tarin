import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { UploadeFileComponent } from './uploade-file.component';

describe('UploadeFileComponent', () => {
  let component: UploadeFileComponent;
  let fixture: ComponentFixture<UploadeFileComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ UploadeFileComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(UploadeFileComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
